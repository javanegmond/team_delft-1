#include <dr_param/param.hpp>
#include <dr_eigen/param.hpp>
#include <dr_eigen/eigen.hpp>
#include <dr_eigen/yaml.hpp>
#include <dr_eigen/ros.hpp>
#include <dr_eigen/tf.hpp>

#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>
#include <moveit/move_group_interface/move_group.h>

#include <geometry_msgs/PoseArray.h>
#include <apc16delft_msgs/InitializeCalibration.h>
#include <apc16delft_msgs/RecordCalibration.h>
#include <apc16delft_msgs/FinalizeCalibration.h>
#include <apc16delft_msgs/GetPose.h>
#include <apc16delft_msgs/ExecuteCalibrationMotion.h>

#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace apc16delft {

/// Perform linear interpolation between two values.
double interpolate(double alpha, double min, double max) {
	return min + alpha * (max - min);
}

/// Perform uniform sampling over a range.
/**
 * Both the min and max values will be included in the sampling if more than one sample is requested.
 */
std::vector<double> sampleLinear(double min, double max, int count) {
	if (count == 0) return {};
	if (count == 1) return {min};

	std::vector<double> result;
	result.reserve(count);
	for (int i = 0; i < count; ++i) {
		result.push_back(min + (max - min) * i / double(count - 1));
	}
	return result;
}

/// Divide the number of samples over a list of radii so that the number of samples if proportional to the surface area of a sphere.
std::vector<int> divideSamples(std::vector<double> const & radii, int total) {
	double sum_squares = 0;
	for (auto const & radius : radii) {
		sum_squares += radius * radius;
	}

	std::vector<int> samples;
	samples.reserve(radii.size());
	for (double radius : radii) {
		samples.push_back(total * radius * radius / sum_squares);
	}
	return samples;
}

/// Calculate the length of an arc between two angles.
/**
 * \param radius The radius of the circle on which the arc lies.
 * \param alpha  The starting angle.
 * \param beta   The end angle.
 */
double arcLength(double radius, double alpha, double beta) {
	return radius * std::abs(beta - alpha);
}

/// Calculate the surface area of a dome.
double domeSurface(double radius, double angle) {
	return 2 * M_PI * radius * radius * (1 - std::cos(angle));
}

/// TODO: comment, height can go from 0 to 1 exclusive.
// Algorithms: http://blog.marmakoide.org/?p=1
Eigen::Vector3d sampleDomePoint(int i, int samples, double height) {
	height = 1 - 2 * height; //interpolate(height, 1, -1);

	double golden_angle = M_PI * (3 - std::sqrt(5));
	double theta = golden_angle * i;
	// original did this for some reason
	//double z = interpolate(i / double(samples - 1), 1 - 1.0 / samples, height);
	double z = -interpolate(i / double(samples - 1), 1, height);
	double radius = std::sqrt(1 - z * z);
	return Eigen::Vector3d{std::cos(theta) * radius, std::sin(theta) * radius, z};
}

class CalibrationNode {
	Eigen::Isometry3d movement_origin = Eigen::Isometry3d::Identity();

	std::string robot_frame;
	std::string fixed_frame;
	std::string attached_frame;
	bool camera_moving;

	/// Total number of samples.
	int samples = 0;
	/// Lowest dome radius to use.
	double radius_min = 0;
	/// Highest dome radius to use.
	double radius_max = 0;
	/// Number of dome radii to use.
	int radius_steps = 0;
	/// The relative height of the dome.
	double dome_height = 0;

	ros::NodeHandle node{"~"};

	struct {
		ros::ServiceClient get_pattern_pose;
		ros::ServiceClient initialize_calibration;
		ros::ServiceClient record_calibration;
		ros::ServiceClient finalize_calibration;
		ros::ServiceClient calibration_motion;
	} services;

	struct {
		ros::Publisher poses;
		ros::Publisher motion_goal;
		ros::Publisher camera_pose;
		ros::Publisher pattern_pose;
		ros::Publisher robot_pose;
		ros::Publisher dome_origin;
		ros::Publisher camera_guess;
		ros::Publisher pattern_guess;
	} publishers;

	struct {
		ros::ServiceServer start;
	} servers;

	tf::TransformListener tf;
	moveit::planning_interface::MoveGroup move_group;

	ros::AsyncSpinner spinner;

	double random_angle;
	std::default_random_engine random_generator;

public:
	CalibrationNode() : move_group(dr::getParam<std::string>(node, "move_group", "manipulator")), spinner(1) {
		robot_frame    = dr::getParam<std::string>(node, "robot_frame");
		fixed_frame    = dr::getParam<std::string>(node, "fixed_frame");
		attached_frame = dr::getParam<std::string>(node, "attached_frame");
		camera_moving  = dr::getParam<bool>(node, "camera_moving");
		samples        = dr::getParam<int>(node, "samples");
		radius_min     = dr::getParam<double>(node, "radius_min");
		radius_max     = dr::getParam<double>(node, "radius_max");
		radius_steps   = dr::getParam<int>(node, "radius_steps");
		dome_height    = dr::getParam<double>(node, "dome_height");
		random_angle   = dr::getParam<double>(node, "random_angle", 15.0 / 180.0 * M_PI);

		publishers.poses                = node.advertise<geometry_msgs::PoseArray>("poses", 1, true);
		publishers.motion_goal          = node.advertise<geometry_msgs::PoseStamped>("motion_goal", 1, true);
		publishers.dome_origin          = node.advertise<geometry_msgs::PoseStamped>("dome_origin", 1, true);
		publishers.camera_pose          = node.advertise<geometry_msgs::PoseStamped>("camera_pose", 1, true);
		publishers.pattern_pose         = node.advertise<geometry_msgs::PoseStamped>("pattern_pose", 1, true);
		publishers.robot_pose           = node.advertise<geometry_msgs::PoseStamped>("robot_pose", 1, true);
		publishers.camera_guess         = node.advertise<geometry_msgs::PoseStamped>("camera_guess", 1, true);
		publishers.pattern_guess        = node.advertise<geometry_msgs::PoseStamped>("pattern_guess", 1, true);

		std::string camera_namespace = dr::getParam<std::string>(node, "camera_namespace");
		services.get_pattern_pose       = node.serviceClient<apc16delft_msgs::GetPose>(camera_namespace + "/get_pattern_pose");
		services.initialize_calibration = node.serviceClient<apc16delft_msgs::InitializeCalibration>(camera_namespace + "/initialize_calibration");
		services.record_calibration     = node.serviceClient<apc16delft_msgs::RecordCalibration>(camera_namespace + "/record_calibration");
		services.finalize_calibration   = node.serviceClient<apc16delft_msgs::FinalizeCalibration>(camera_namespace + "/finalize_calibration");
		services.calibration_motion     = node.serviceClient<apc16delft_msgs::ExecuteCalibrationMotion>("/motion_executor/execute_calibration_motion");
		
		spinner.start();

		servers.start                   = node.advertiseService("do_calibration", &CalibrationNode::onDoCalibration, this);

		ROS_INFO_STREAM("Node initialized.");
	}

private:
	bool onDoCalibration(std_srvs::Empty::Request &, std_srvs::Empty::Response &) {
		return doCalibration();
	}

	bool doCalibration() {
		// Find the current pose of the pattern with respect to the camera.
		boost::optional<Eigen::Isometry3d> pattern = getPatternPose();
		if (!pattern) {
			ROS_ERROR_STREAM("Camera failed to find pattern pose.");
			return false;
		}

		Eigen::Isometry3d camera_guess;
		Eigen::Isometry3d pattern_guess;
		Eigen::Isometry3d dome_origin;

		ros::Time now = ros::Time::now();

		// Guess the initial camera and pattern poses based on the found pattern and the pose of the attached frame (camera or pattern).
		// Use the non-attached frame as dome origin.
		Eigen::Isometry3d attached_to_robot = lookupTransform(robot_frame, attached_frame, now);
		if (camera_moving) {
			camera_guess  = attached_to_robot;
			pattern_guess = lookupTransform(fixed_frame, attached_frame, now) * (*pattern);
			dome_origin   = pattern_guess;
		} else {
			camera_guess  = lookupTransform(fixed_frame, attached_frame, now) * pattern->inverse();
			pattern_guess = attached_to_robot;
			dome_origin   = camera_guess;
		}

		publishers.dome_origin.publish(dr::toRosPoseStamped(dome_origin, fixed_frame, ros::Time::now()));
		publishers.camera_guess.publish(dr::toRosPoseStamped(camera_guess, camera_moving ? robot_frame : fixed_frame, now));
		publishers.pattern_guess.publish(dr::toRosPoseStamped(pattern_guess, camera_moving ? fixed_frame : robot_frame, now));

		// Initialize the camera calibration.
		if (!initializeCalibration(fixed_frame, robot_frame, camera_moving, camera_guess, pattern_guess)) {
			ROS_ERROR_STREAM("Initialization of camera calibration failed.");
			return false;
		}

		// Generate robot poses.
		std::vector<Eigen::Isometry3d> poses = generateRobotPoses(dome_origin);
		publishPoses(poses, fixed_frame);

		//ROS_INFO_STREAM("Showing poses, abort now if you wish");
		//ros::Duration(10.0).sleep();

		int failed = 0;
		int sample = 0;
		ROS_INFO_STREAM("Number of poses: " << poses.size());
		// Move to each pose in turn and record data.
		for (Eigen::Isometry3d const & pose : poses) {
			++sample;
			ROS_INFO_STREAM("Attempting to record sample " << sample << "/" << samples << ", so far " << failed << " samples failed.");

			if (!node.ok()) {
				ROS_WARN_STREAM("Calibration interrupted by user.");
				return false;
			}
			ROS_INFO_STREAM("Moving to pose:\n" << dr::toYaml(pose));
			if (!moveToPose(pose)) {
				++failed;
				ROS_ERROR_STREAM("Failed to move to camera pose. " << failed << " samples failed.");
				continue;
			}
			ros::Duration(0.5).sleep();

			Eigen::Isometry3d robot_pose = lookupTransform(fixed_frame, robot_frame, ros::Time::now());
			publishers.robot_pose.publish(dr::toRosPoseStamped(robot_pose, fixed_frame, ros::Time::now()));

			if (!recordCalibration(robot_pose)) {
				++failed;
				ROS_ERROR_STREAM("Failed to record calibration pattern. " << failed << " samples failed.");
				continue;
			}
		}

		// Finalize the calibration.
		ROS_INFO_STREAM("Finalizing calibration with " << (samples - failed) << "/" << samples << " samples.");
		boost::optional<apc16delft_msgs::FinalizeCalibration::Response> result = finalizeCalibration();
		if (!result) {
			ROS_ERROR_STREAM("Calibration finalization failed.");
			return false;
		}

		// Print the result.
		ROS_INFO_STREAM("Calibration succeeded with a reprojection error of " << result->reprojection_error << ".");
		ROS_INFO_STREAM("Camera pose with respect to `" << result->camera_pose.header.frame_id << "':\n" << result->camera_pose.pose);
		publishers.camera_pose.publish(result->camera_pose);
		ROS_INFO_STREAM("Pattern pose with respect to `" << result->pattern_pose.header.frame_id << "':\n" << result->pattern_pose.pose);
		Eigen::AngleAxisd angle_axis{dr::toEigen(result->pattern_pose.pose).rotation()};
		ROS_INFO_STREAM("Axis angle: " << dr::toYaml(angle_axis.axis()) << " " << angle_axis.angle());
		publishers.pattern_pose.publish(result->pattern_pose);


		return true;
	}

	/// Lookup a transform between two frames as an Eigen isometry.
	Eigen::Isometry3d lookupTransform(std::string const & target_frame, std::string const & source_frame, ros::Time const & time) {
		tf::StampedTransform transform;
		if (!tf.waitForTransform(target_frame, source_frame, time, ros::Duration(1))) {
			throw std::runtime_error("Transform from " + source_frame + " to " + target_frame + " not available.");
		}
		tf.lookupTransform(target_frame, source_frame, time, transform);
		return dr::toEigen(transform);
	}

	/// Publish a list of poses for debugging purposes.
	void publishPoses(std::vector<Eigen::Isometry3d> const & poses, std::string const & frame, ros::Time time = ros::Time::now()) {
		geometry_msgs::PoseArray message;
		message.header.frame_id = frame;
		message.header.stamp    = time;
		message.poses.reserve(poses.size());

		for (Eigen::Isometry3d const & pose : poses) {
			message.poses.push_back(dr::toRosPose(pose));
		}

		publishers.poses.publish(message);
	}

	/// Calculate a pose for a point on a sphere.
	/**
	 * The Z axis will point towards the sphere origin.
	 */
	Eigen::Isometry3d robotPose(Eigen::Vector3d const & sphere_point) {
		Eigen::Matrix3d rotation;
		rotation.col(2) = -1 * sphere_point.normalized();
		rotation.col(0) = dr::axes::x().cross(rotation.col(2)).normalized();
		rotation.col(1) = rotation.col(2).cross(rotation.col(0)).normalized();
		return Eigen::Translation3d(sphere_point) * Eigen::Quaterniond{rotation} * dr::rotateY(camera_moving ? 0 : M_PI);
	}

	/// Generaterobot poses on a number of domes.
	std::vector<Eigen::Isometry3d> generateRobotPoses(Eigen::Isometry3d const & origin) {
		std::vector<double> radii      = sampleLinear(radius_min, radius_max, radius_steps);
		std::vector<int> radii_samples = divideSamples(radii, samples);

		std::vector<Eigen::Isometry3d> result;
		std::uniform_real_distribution<double> rd(-random_angle, random_angle);

		for (std::size_t radius_i = 0; radius_i < radii.size(); ++radius_i) {
			double radius = radii[radius_i];
			int dome_samples = radii_samples[radius_i];
			for (int i = 0; i < dome_samples; ++i) {
				Eigen::Vector3d point = sampleDomePoint(i, dome_samples, dome_height) * radius;

				// generate random angle perturbation
				Eigen::AngleAxisd random_angle(rd(random_generator),
					Eigen::Vector3d(rd(random_generator), rd(random_generator), rd(random_generator)).normalized());

				result.push_back(origin * robotPose(point) * dr::rotateZ(1.0 * M_PI) * random_angle);
				//result.push_back(origin * robotPose(point) *  random_angle);
			}
		}

		return result;
	}

	/// Move the robot to a given pose.
	bool moveToPose(Eigen::Isometry3d pose) {
		pose = pose * lookupTransform(attached_frame, "ee_link", ros::Time::now());
		apc16delft_msgs::ExecuteCalibrationMotion service;
		service.request.calibration_pose = dr::toRosPoseStamped(pose, fixed_frame, ros::Time::now());
		publishers.motion_goal.publish(dr::toRosPoseStamped(pose, fixed_frame, ros::Time::now()));

	//	return services.calibration_motion.call(service);
		return doMove(service.request);


//		moveit::planning_interface::MoveGroup::Plan plan;
//		move_group.setStartStateToCurrentState();
//		move_group.setPoseTarget(pose, "gripper_tool0");
//		moveit::planning_interface::MoveItErrorCode error;
//
//		ROS_INFO_STREAM("Planning path.");
//		error = move_group.plan(plan);
//		if (!error) {
//			ROS_ERROR_STREAM("Failed to plan trajectory to target pose: " << error);
//			return false;
//		}
//
//		ROS_INFO_STREAM("Exucting plan.");
//		error = move_group.execute(plan);
//		if (!error) {
//			ROS_ERROR_STREAM("Failed to execute trajectory: " << error);
//			return false;
//		}
//
//		return true;
	}

	bool doMove(apc16delft_msgs::ExecuteCalibrationMotion::Request & req) {

		moveit::planning_interface::MoveGroup current_group("manipulator");
		current_group.clearPoseTargets();
		
		std::vector<double> joint_value_current;
		std::vector<double> joint_value_target;	
		
	//	ROS_INFO_STREAM("Getting current state");

		robot_state::RobotStatePtr current_state(current_group.getCurrentState());
		const robot_state::JointModelGroup *jmg = current_state->getJointModelGroup(current_group.getName());

		current_state->copyJointGroupPositions(jmg, joint_value_current);

	//	ROS_INFO_STREAM("Setting target state");

		const robot_state::RobotState target_state = current_group.getJointValueTarget();

		current_group.setJointValueTarget(req.calibration_pose, "ee_link");
	//	ROS_INFO_STREAM("Setting velocity scaling");
//		current_group.setMaxVelocityScalingFactor(0.5);
		moveit::planning_interface::MoveGroup::Plan calibration_motion_plan;

	//	ROS_INFO_STREAM("Setting current state");
		current_group.setStartState(*current_state);
		current_group.setPlannerId("RRTConnectkConfigDefault");
//		current_group.setPlannerId("LBKPIECEkConfigDefault");

		current_group.allowReplanning(true);
		current_group.setNumPlanningAttempts(5);
		
	//	ROS_INFO_STREAM("Start planning!");
		if(current_group.plan(calibration_motion_plan) != 1) {
			ROS_ERROR_STREAM("No motion plan found.");
			return false;
		}
	//	ROS_INFO_STREAM("Sanity checking calibration motion");
		//std::cout << calibration_motion_plan.trajectory_.joint_trajectory << std::endl;
		if(!checkCalibrationTrajectorySanity(calibration_motion_plan.trajectory_.joint_trajectory, &current_group)) {
			ROS_ERROR_STREAM("Sanity check violated.");
			return false;
		}
//		ROS_INFO_STREAM("Sanity check passed");
	//	std::cout << calibration_motion_plan.trajectory_.joint_trajectory << std::endl;
	//	ros::Duration(5.0).sleep();
		ROS_INFO_STREAM("Executing calibration motion.");
		current_group.execute(calibration_motion_plan);
		return true;
	}

bool checkCalibrationTrajectorySanity(trajectory_msgs::JointTrajectory & motion_trajectory, moveit::planning_interface::MoveGroup* current_group) {
	std::vector<double> current_joint_values;
	double *joint_value_ptr;
	double calibration_trajectory_tolerance_ = 3.20;
	//trajectory_msgs::JointTrajectoryPoint starting_point = motion_trajectory.points[0];

	robot_state::RobotStatePtr kinematic_state(current_group->getCurrentState());
	//Get current joint values.
	double joint_diff;
	double distance = 0.0;
	joint_value_ptr = kinematic_state->getVariablePositions();
	const std::vector<std::string> joint_names= kinematic_state->getVariableNames();
	//Compute distance between current state and the trajectory start state
	std::vector<std::string>::iterator it;
	for (size_t traj_idx = 0; traj_idx < motion_trajectory.points.size(); traj_idx++) {
		for (size_t idx=0; idx < motion_trajectory.points[traj_idx].positions.size(); idx++) {
			it = std::find(motion_trajectory.joint_names.begin(), motion_trajectory.joint_names.end(),joint_names[idx]);
			int pos_idx = std::distance(motion_trajectory.joint_names.begin(), it);

			joint_diff = joint_value_ptr[idx] - motion_trajectory.points[traj_idx].positions[pos_idx];
			//Adjust starting point of cached trajectory to current position for practical reasons.
//			motion_trajectory.points[0].positions[pos_idx] = joint_value_ptr[idx];
			ROS_DEBUG_STREAM(joint_names[idx] <<": " << joint_value_ptr[idx]);
			distance += (joint_diff*joint_diff);
		}
		distance = sqrt(distance);
		if (distance > calibration_trajectory_tolerance_) {
			ROS_ERROR_STREAM("Motion safety check violation, distance is " << distance << " which is above the threshold of " << calibration_trajectory_tolerance_ << ".");
			return false;
			break;
		}
	}
	
	return true;

}
	/// Get the pose of a calibration pattern.
	boost::optional<Eigen::Isometry3d> getPatternPose() {
		apc16delft_msgs::GetPose service;
		if (!services.get_pattern_pose.call(service)) {
			return boost::none;
		}

		return dr::toEigen(service.response.data);
	}

	/// Initialize the camera calibration.
	bool initializeCalibration(
		std::string const & static_frame,
		std::string const & robot_frame,
		bool camera_moving,
		boost::optional<Eigen::Isometry3d> const & camera_guess = {},
		boost::optional<Eigen::Isometry3d> const & pattern_guess = {}
	) {
		apc16delft_msgs::InitializeCalibration service;
		service.request.static_frame = static_frame;
		service.request.moving_frame = robot_frame;
		service.request.camera_moving = camera_moving;
		if (camera_guess)  service.request.camera_guess  = dr::toRosPose(*camera_guess);
		if (pattern_guess) service.request.pattern_guess = dr::toRosPose(*pattern_guess);
		return services.initialize_calibration.call(service);
	}

	/// Record a sample for the camera calibration.
	bool recordCalibration(
		Eigen::Isometry3d const & robot_pose
	) {
		apc16delft_msgs::RecordCalibration service;
		service.request.pose = dr::toRosPose(robot_pose);
		return services.record_calibration.call(service);
	}

	/// Finalize the camera calibration.
	boost::optional<apc16delft_msgs::FinalizeCalibration::Response> finalizeCalibration() {
		apc16delft_msgs::FinalizeCalibration service;
		if (!services.finalize_calibration.call(service)) {
			return boost::none;
		}

		return service.response;
	}

};

}

int main(int argc, char * * argv) {
	ros::init(argc, argv, ROS_PACKAGE_NAME);
	apc16delft::CalibrationNode node;

	ros::Rate loop_rate(1000);
	while(ros::ok()) {
		ros::spinOnce();
		loop_rate.sleep();
	}	
	ros::waitForShutdown();
	return 0;
}	
