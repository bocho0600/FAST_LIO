/// @file global_localization.cpp
/// @brief Re-localise FAST-LIO's odometry inside a prior map, by ICP.
///
/// A C++ port of the approach in HViktorTsoi/FAST_LIO_LOCALIZATION, which is ROS 1 and
/// depends on Python 2.7 plus Open3D. Neither is available here, so the registration is
/// rebuilt on PCL, which this package already links. The algorithm is otherwise the same:
/// low-frequency global localisation (0.5 Hz by default) supplying map -> odom, fused with
/// FAST-LIO's high-rate odom -> base_link, so the expensive part runs rarely and the pose
/// stays high-rate.
///
/// Deliberately a separate executable rather than a second timer inside LaserMappingNode.
/// That node is spun with rclcpp::spin -- a single-threaded executor -- so an ICP taking
/// hundreds of milliseconds would block its 100 Hz timer_callback and drop scans. The
/// filter's real-time path must not be able to stall behind registration.
///
/// What this does NOT do: correct FAST-LIO's internal state. It publishes map -> odom, so
/// drift accumulated in odom is absorbed by that edge rather than removed. The odom frame
/// stays continuous, which is what nav2 wants; map -> base_link is the drift-free pose.
///
/// Disabled unless relocalization.enable is true, in which case a prior map is required.

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fast_lio
{
using PointT = pcl::PointXYZI;
using Cloud = pcl::PointCloud<PointT>;

/// Expand a leading ~/ the way the mapping node's map_file_path does, so the same path
/// string works in both configs.
static std::string expand_user(const std::string &path)
{
    if (path.size() < 2 || path[0] != '~' || path[1] != '/')
        return path;
    const char *home = std::getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
}

class GlobalLocalizationNode : public rclcpp::Node
{
public:
    explicit GlobalLocalizationNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : rclcpp::Node("global_localization", options)
    {
        enable_ = this->declare_parameter<bool>("relocalization.enable", false);
        prior_map_path_ = this->declare_parameter<std::string>("relocalization.prior_map_path", "");
        map_voxel_size_ = this->declare_parameter<double>("relocalization.map_voxel_size", 0.4);
        scan_voxel_size_ = this->declare_parameter<double>("relocalization.scan_voxel_size", 0.1);
        freq_hz_ = this->declare_parameter<double>("relocalization.freq_hz", 0.5);
        fitness_threshold_ = this->declare_parameter<double>("relocalization.fitness_threshold", 0.95);
        fov_rad_ = this->declare_parameter<double>("relocalization.fov_rad", 1.6);
        fov_far_ = this->declare_parameter<double>("relocalization.fov_far", 150.0);
        coarse_scale_ = this->declare_parameter<double>("relocalization.coarse_scale", 5.0);
        // Upstream hardcodes 1.0 * scale. Exposed because it is the gate's real sensitivity:
        // fitness counts a point as an inlier if a map point is within this distance, so at
        // 1.0 m a half-metre misalignment still scores near 1.0 and passes fitness_threshold.
        // Tighten it towards the map voxel size to make the threshold mean something.
        max_corr_dist_ = this->declare_parameter<double>(
            "relocalization.max_correspondence_distance", 1.0);
        max_iterations_ = this->declare_parameter<int>("relocalization.max_iterations", 20);
        tf_rate_hz_ = this->declare_parameter<double>("relocalization.tf_publish_rate_hz", 10.0);
        map_frame_ = this->declare_parameter<std::string>("relocalization.map_frame", "map");
        odom_frame_ = this->declare_parameter<std::string>("relocalization.odom_frame", "odom");
        scan_topic_ = this->declare_parameter<std::string>("relocalization.scan_topic", "/cloud_registered");
        odom_topic_ = this->declare_parameter<std::string>("relocalization.odom_topic", "/Odometry");
        // Optional bootstrap without RViz: [x, y, z, roll, pitch, yaw] of base_link in the map
        // frame. Beyond upstream, which only ever accepts /initialpose -- a robot in the field
        // has no one to click for it.
        initial_pose_ = this->declare_parameter<std::vector<double>>("relocalization.initial_pose",
                                                                     std::vector<double>{});

        if (!enable_)
        {
            RCLCPP_INFO(this->get_logger(),
                        "relocalization.enable is false: idle, publishing no %s -> %s.",
                        map_frame_.c_str(), odom_frame_.c_str());
            return;
        }

        if (!load_prior_map())
            return;  // load_prior_map() has already logged and cleared enable_

        // Two callback groups so the TF republisher keeps running while an ICP is in flight.
        // With one group (or a single-threaded executor) map -> odom would freeze for the
        // duration of every registration.
        icp_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        io_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions io_opts;
        io_opts.callback_group = io_group_;

        scan_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            scan_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_scan_ = msg;
            },
            io_opts);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 20,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_odom_ = msg;
            },
            io_opts);

        initialpose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 1,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
                Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
                pose_to_matrix(msg->pose.pose, guess);
                seed_from_base_pose(guess, "/initialpose");
            },
            io_opts);

        map_to_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/map_to_odom", 10);
        submap_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/submap", 1);
        scan_in_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cur_scan_in_map", 1);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        if (initial_pose_.size() == 6)
        {
            Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
            rpy_xyz_to_matrix(initial_pose_, guess);
            seed_from_base_pose(guess, "relocalization.initial_pose");
        }
        else if (!initial_pose_.empty())
        {
            RCLCPP_WARN(this->get_logger(),
                        "relocalization.initial_pose has %zu values, expected 6 "
                        "[x y z roll pitch yaw]; ignoring and waiting for /initialpose.",
                        initial_pose_.size());
        }

        icp_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(freq_hz_, 1e-3)),
            std::bind(&GlobalLocalizationNode::localization_callback, this), icp_group_);
        tf_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(tf_rate_hz_, 1e-3)),
            std::bind(&GlobalLocalizationNode::publish_map_to_odom, this), io_group_);

        RCLCPP_INFO(this->get_logger(),
                    "Re-localisation enabled: %zu map points, %.2f Hz ICP, fitness > %.2f. "
                    "Waiting for an initial pose guess.",
                    global_map_->size(), freq_hz_, fitness_threshold_);
    }

private:
    /// Load and downsample the prior map. Clears enable_ and returns false on any failure --
    /// a missing map must not leave the node silently publishing an identity transform.
    bool load_prior_map()
    {
        const std::string path = expand_user(prior_map_path_);
        if (path.empty())
        {
            RCLCPP_ERROR(this->get_logger(),
                         "relocalization.enable is true but relocalization.prior_map_path is "
                         "empty. Nothing to localise against; staying idle.");
            enable_ = false;
            return false;
        }

        auto raw = std::make_shared<Cloud>();
        if (pcl::io::loadPCDFile<PointT>(path, *raw) == -1 || raw->empty())
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Could not read a prior map from '%s'. Staying idle.", path.c_str());
            enable_ = false;
            return false;
        }

        global_map_ = voxel_downsample(raw, map_voxel_size_);
        RCLCPP_INFO(this->get_logger(), "Prior map '%s': %zu points -> %zu after %.2f m voxels.",
                    path.c_str(), raw->size(), global_map_->size(), map_voxel_size_);
        return true;
    }

    static Cloud::Ptr voxel_downsample(const Cloud::ConstPtr &in, double leaf)
    {
        auto out = std::make_shared<Cloud>();
        if (leaf <= 0.0)
        {
            *out = *in;
            return out;
        }
        pcl::VoxelGrid<PointT> grid;
        grid.setInputCloud(in);
        grid.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
        grid.filter(*out);
        return out;
    }

    static void pose_to_matrix(const geometry_msgs::msg::Pose &pose, Eigen::Matrix4f &out)
    {
        const Eigen::Quaternionf q(static_cast<float>(pose.orientation.w),
                                   static_cast<float>(pose.orientation.x),
                                   static_cast<float>(pose.orientation.y),
                                   static_cast<float>(pose.orientation.z));
        out.setIdentity();
        out.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        out(0, 3) = static_cast<float>(pose.position.x);
        out(1, 3) = static_cast<float>(pose.position.y);
        out(2, 3) = static_cast<float>(pose.position.z);
    }

    static void rpy_xyz_to_matrix(const std::vector<double> &v, Eigen::Matrix4f &out)
    {
        tf2::Quaternion q;
        q.setRPY(v[3], v[4], v[5]);
        geometry_msgs::msg::Pose pose;
        pose.position.x = v[0];
        pose.position.y = v[1];
        pose.position.z = v[2];
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
        pose_to_matrix(pose, out);
    }

    /// Turn a guessed base_link-in-map pose into the map -> odom seed.
    ///
    /// Upstream feeds /initialpose straight into ICP as the map -> odom initial value, which
    /// is only right while odom -> base_link is still near identity, i.e. immediately after
    /// startup. Composing out the current odometry makes a guess supplied mid-run work too:
    ///   T_map_odom = T_map_base * inverse(T_odom_base)
    void seed_from_base_pose(const Eigen::Matrix4f &T_map_base, const std::string &source)
    {
        Eigen::Matrix4f T_odom_base = Eigen::Matrix4f::Identity();
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (latest_odom_)
                pose_to_matrix(latest_odom_->pose.pose, T_odom_base);
        }
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            T_map_to_odom_ = T_map_base * invert_rigid(T_odom_base);
            seeded_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "Seeded map -> odom from %s.", source.c_str());
    }

    /// Inverse of a rigid transform, without the general matrix inverse.
    static Eigen::Matrix4f invert_rigid(const Eigen::Matrix4f &T)
    {
        Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
        const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
        out.block<3, 3>(0, 0) = R.transpose();
        out.block<3, 1>(0, 3) = -R.transpose() * T.block<3, 1>(0, 3);
        return out;
    }

    /// Keep only the map points the sensor could plausibly see from T_map_base, tested in the
    /// base frame but returned in the map frame -- ICP needs them in map coordinates.
    Cloud::Ptr crop_global_map_in_fov(const Eigen::Matrix4f &T_map_base) const
    {
        const Eigen::Matrix4f T_base_map = invert_rigid(T_map_base);
        const bool omnidirectional = fov_rad_ > M_PI;
        const double half_fov = fov_rad_ / 2.0;

        auto out = std::make_shared<Cloud>();
        out->reserve(global_map_->size());
        for (const auto &p : global_map_->points)
        {
            const Eigen::Vector4f in_base =
                T_base_map * Eigen::Vector4f(p.x, p.y, p.z, 1.0f);
            if (in_base.x() >= static_cast<float>(fov_far_))
                continue;
            if (!omnidirectional && in_base.x() <= 0.0f)
                continue;
            if (std::fabs(std::atan2(in_base.y(), in_base.x())) >= half_fov)
                continue;
            out->push_back(p);
        }
        out->width = out->size();
        out->height = 1;
        out->is_dense = false;
        return out;
    }

    /// Fraction of source points with a target point inside max_corr.
    ///
    /// This reimplements Open3D's `fitness`, which upstream's 0.95 threshold refers to.
    /// PCL's getFitnessScore() is NOT the same quantity -- it is a mean squared distance, so
    /// lower is better and comparing it against 0.95 would invert the test.
    static double inlier_fraction(const Cloud::ConstPtr &source, const Cloud::ConstPtr &target,
                                  double max_corr)
    {
        if (source->empty() || target->empty())
            return 0.0;
        pcl::KdTreeFLANN<PointT> tree;
        tree.setInputCloud(target);
        const float max_sq = static_cast<float>(max_corr * max_corr);
        std::vector<int> idx(1);
        std::vector<float> sq_dist(1);
        std::size_t inliers = 0;
        for (const auto &p : source->points)
        {
            if (tree.nearestKSearch(p, 1, idx, sq_dist) > 0 && sq_dist[0] <= max_sq)
                ++inliers;
        }
        return static_cast<double>(inliers) / static_cast<double>(source->size());
    }

    /// One ICP pass at `scale`. Mirrors upstream registration_at_scale(): both clouds are
    /// downsampled by their voxel size times the scale, and the correspondence distance is
    /// 1.0 * scale, so a coarse pass sees a blurrier cloud and searches further.
    bool registration_at_scale(const Cloud::ConstPtr &scan, const Cloud::ConstPtr &map,
                               const Eigen::Matrix4f &initial, double scale,
                               Eigen::Matrix4f &result, double &fitness) const
    {
        const Cloud::Ptr scan_ds = voxel_downsample(scan, scan_voxel_size_ * scale);
        const Cloud::Ptr map_ds = voxel_downsample(map, map_voxel_size_ * scale);
        if (scan_ds->empty() || map_ds->empty())
        {
            fitness = 0.0;
            return false;
        }

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(scan_ds);
        icp.setInputTarget(map_ds);
        icp.setMaxCorrespondenceDistance(max_corr_dist_ * scale);
        icp.setMaximumIterations(max_iterations_);

        Cloud aligned;
        icp.align(aligned, initial);
        if (!icp.hasConverged())
        {
            fitness = 0.0;
            return false;
        }
        result = icp.getFinalTransformation();

        auto transformed = std::make_shared<Cloud>();
        pcl::transformPointCloud(*scan_ds, *transformed, result);
        fitness = inlier_fraction(transformed, map_ds, max_corr_dist_ * scale);
        return true;
    }

    void localization_callback()
    {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
        nav_msgs::msg::Odometry::ConstSharedPtr odom_msg;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            scan_msg = latest_scan_;
            odom_msg = latest_odom_;
        }

        Eigen::Matrix4f T_map_odom;
        bool seeded;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            T_map_odom = T_map_to_odom_;
            seeded = seeded_;
        }

        if (!seeded)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                 "No initial pose yet: publish /initialpose or set "
                                 "relocalization.initial_pose.");
            return;
        }
        if (!scan_msg || !odom_msg)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                 "Waiting for %s and %s.", scan_topic_.c_str(),
                                 odom_topic_.c_str());
            return;
        }

        auto scan = std::make_shared<Cloud>();
        pcl::fromROSMsg(*scan_msg, *scan);
        if (scan->empty())
            return;

        Eigen::Matrix4f T_odom_base = Eigen::Matrix4f::Identity();
        pose_to_matrix(odom_msg->pose.pose, T_odom_base);
        const Cloud::Ptr submap = crop_global_map_in_fov(T_map_odom * T_odom_base);
        if (submap->empty())
        {
            RCLCPP_WARN(this->get_logger(),
                        "No prior-map points in view: the guess is far from the map, or "
                        "relocalization.fov_far/%.2f fov_rad/%.2f are too tight.",
                        fov_far_, fov_rad_);
            return;
        }

        // Coarse then fine, exactly as upstream: scale 5 to pull in a bad guess, then scale 1
        // seeded with that result. One pass at scale 1 alone converges to a local minimum for
        // anything but a good initial guess.
        Eigen::Matrix4f T_coarse = T_map_odom;
        double fitness_coarse = 0.0;
        if (!registration_at_scale(scan, submap, T_map_odom, coarse_scale_, T_coarse, fitness_coarse))
        {
            RCLCPP_WARN(this->get_logger(), "Coarse ICP did not converge.");
            return;
        }

        Eigen::Matrix4f T_fine = T_coarse;
        double fitness_fine = 0.0;
        if (!registration_at_scale(scan, submap, T_coarse, 1.0, T_fine, fitness_fine))
        {
            RCLCPP_WARN(this->get_logger(), "Fine ICP did not converge.");
            return;
        }

        if (fitness_fine <= fitness_threshold_)
        {
            RCLCPP_WARN(this->get_logger(),
                        "Rejected: fitness %.3f <= %.3f (coarse %.3f). Keeping the previous "
                        "%s -> %s.",
                        fitness_fine, fitness_threshold_, fitness_coarse, map_frame_.c_str(),
                        odom_frame_.c_str());
            return;
        }

        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            T_map_to_odom_ = T_fine;
            localized_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "Re-localised: fitness %.3f over %zu submap points.",
                    fitness_fine, submap->size());

        if (submap_pub_->get_subscription_count() > 0)
            publish_cloud(submap_pub_, submap, scan_msg->header.stamp);
        if (scan_in_map_pub_->get_subscription_count() > 0)
        {
            auto in_map = std::make_shared<Cloud>();
            pcl::transformPointCloud(*scan, *in_map, T_fine);
            publish_cloud(scan_in_map_pub_, in_map, scan_msg->header.stamp);
        }
    }

    void publish_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub,
                       const Cloud::ConstPtr &cloud, const builtin_interfaces::msg::Time &stamp)
    {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.stamp = stamp;
        msg.header.frame_id = map_frame_;
        pub->publish(msg);
    }

    /// Republish map -> odom faster than ICP runs, so consumers always see a fresh stamp
    /// rather than a transform that goes stale between registrations. This is the job
    /// upstream splits into a second script (transform_fusion.py); there is no reason for a
    /// separate process here.
    void publish_map_to_odom()
    {
        Eigen::Matrix4f T;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            if (!localized_)
                return;
            T = T_map_to_odom_;
        }

        const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
        const Eigen::Quaternionf q(R);
        const auto stamp = this->get_clock()->now();

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = map_frame_;
        tf_msg.child_frame_id = odom_frame_;
        tf_msg.transform.translation.x = T(0, 3);
        tf_msg.transform.translation.y = T(1, 3);
        tf_msg.transform.translation.z = T(2, 3);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(tf_msg);

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = map_frame_;
        odom_msg.child_frame_id = odom_frame_;
        odom_msg.pose.pose.position.x = T(0, 3);
        odom_msg.pose.pose.position.y = T(1, 3);
        odom_msg.pose.pose.position.z = T(2, 3);
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();
        map_to_odom_pub_->publish(odom_msg);
    }

    bool enable_ = false;
    std::string prior_map_path_, map_frame_, odom_frame_, scan_topic_, odom_topic_;
    double map_voxel_size_ = 0.4, scan_voxel_size_ = 0.1, freq_hz_ = 0.5;
    double fitness_threshold_ = 0.95, fov_rad_ = 1.6, fov_far_ = 150.0;
    double coarse_scale_ = 5.0, tf_rate_hz_ = 10.0, max_corr_dist_ = 1.0;
    int max_iterations_ = 20;
    std::vector<double> initial_pose_;

    Cloud::Ptr global_map_;

    mutable std::mutex data_mutex_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_scan_;
    nav_msgs::msg::Odometry::ConstSharedPtr latest_odom_;

    mutable std::mutex pose_mutex_;
    Eigen::Matrix4f T_map_to_odom_ = Eigen::Matrix4f::Identity();
    bool seeded_ = false;    // an initial guess has been supplied
    bool localized_ = false; // at least one registration has been accepted

    rclcpp::CallbackGroup::SharedPtr icp_group_, io_group_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr map_to_odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr submap_pub_, scan_in_map_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr icp_timer_, tf_timer_;
};

} // namespace fast_lio

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    // Multi-threaded so the TF republisher is not blocked behind an ICP.
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<fast_lio::GlobalLocalizationNode>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
