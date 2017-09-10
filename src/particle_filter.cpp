/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 20;
	// create normal (Gaussian) distributions
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle curr_particle;
    curr_particle.id = i;
    curr_particle.x = dist_x(gen);
    curr_particle.y = dist_y(gen);
    curr_particle.theta = dist_theta(gen);
    curr_particle.weight = 1.0;
    weights.push_back(curr_particle.weight);
    particles.push_back(curr_particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  double vel_yr = velocity / yaw_rate;
  double yr_dt = yaw_rate * delta_t;
  double vel_dt = velocity * delta_t;

  if (fabs(yaw_rate) >= 0.0001) {
    // if turning angle is significant enough
    // calculate circular motion
    for (int i = 0; i < num_particles; i++) {
      double x = particles[i].x;
      double y = particles[i].y;
      double theta = particles[i].theta;
      double new_theta = theta + yr_dt;
      double new_x = x + (vel_yr * (sin(new_theta) - sin(theta)));
      double new_y = y + (vel_yr * (cos(theta) - cos(new_theta)));
      particles[i].x = new_x + dist_x(gen);
      particles[i].y = new_y + dist_y(gen);
      particles[i].theta = new_theta + dist_theta(gen);
    }
  } else {
    // approximate for straight motion
    for (int i = 0; i < num_particles; i++) {
      double x = particles[i].x;
      double y = particles[i].y;
      double theta = particles[i].theta;
      double new_x = x + (vel_dt * cos(theta));
      double new_y = y + (vel_dt * sin(theta));
      particles[i].x = new_x + dist_x(gen);
      particles[i].y = new_y + dist_y(gen);
    }
  }

}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i++) {
    double min_distance = INFINITY;
    int min_id = observations[i].id;
    for (int j = 0; j < predicted.size(); j++) {
      Map::single_landmark_s curr_predicted = predicted[j];
      double curr_distance = sqrt(pow(observations[i].x - curr_predicted.x_f, 2.0) + pow(observations[i].y - curr_predicted.y_f, 2.0));
      if (curr_distance < min_distance) {
        min_distance = curr_distance;
        // account for landmarks being numbered starting from 1
        min_id = curr_predicted.id_i - 1;
      }
    }
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // multivariate gaussian probability density function product normalization factor
  double gauss_norm = 1.0 / (2.0*M_PI * std_landmark[0] * std_landmark[1]);
  double std_land_x_2 = 2.0 * pow(std_landmark[0], 2.0);
  double std_land_y_2 = 2.0 * pow(std_landmark[1], 2.0);

  for (int i = 0; i < num_particles; i++) {
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    double cos_theta = cos(particles[i].theta);
    double sin_theta = sin(particles[i].theta);

    // create list of observations translated to map coordinates
    vector<LandmarkObs> map_observations;
    for (int j = 0; j < observations.size(); j++) {
        LandmarkObs map_obs;
        // transform to map coordinates
        map_obs.x = particles[i].x + (observations[j].x * cos_theta) - (observations[j].y * sin_theta);
        map_obs.y = particles[i].y + (observations[j].x * sin_theta) + (observations[j].y * cos_theta);
        map_observations.push_back(map_obs);
    }

    // create list of landmarks in range
    vector<Map::single_landmark_s> landmarks_in_range;
    for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
      double lm_distance = sqrt(pow(map_landmarks.landmark_list[k].x_f - particles[i].x, 2.0) + pow(map_landmarks.landmark_list[k].y_f - particles[i].y, 2.0));
      if (lm_distance <= sensor_range) {
        landmarks_in_range.push_back(map_landmarks.landmark_list[k]);
      }
    }

    dataAssociation(landmarks_in_range, map_observations);

    // calculate weights and add associations to particle
    double curr_weight = 1.0;
    for (int j = 0; j < map_observations.size(); j++) {
      Map::single_landmark_s nearest_landmark = map_landmarks.landmark_list[map_observations[j].id];

      associations.push_back(nearest_landmark.id_i);
      sense_x.push_back(map_observations[j].x);
      sense_y.push_back(map_observations[j].y);

      // multivariate gaussian probability density function product
      double exponent = -1 * (pow(map_observations[j].x - nearest_landmark.x_f, 2.0) / (std_land_x_2) +
                        pow(map_observations[j].y - nearest_landmark.y_f, 2.0) / (std_land_y_2));
      curr_weight *= (gauss_norm * exp(exponent));
    }

    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
    particles[i].weight = curr_weight;
    weights[i] = curr_weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> resampled_particles;
  default_random_engine gen;
  discrete_distribution<> dist(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles[dist(gen)]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
