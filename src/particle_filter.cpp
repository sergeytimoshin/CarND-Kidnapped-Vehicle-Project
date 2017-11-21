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
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    particles.resize(num_particles);
    weights.resize(num_particles);

    random_device rd;
    default_random_engine gen(rd());

    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (auto &p: particles) {
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }

    for (auto &w: weights) {
       w = 1;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    random_device rd;
    default_random_engine gen(rd());

    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (auto &p: particles) {
        // Prevent division by zero for yaw rate
        if (abs(yaw_rate) > 0.0001) {
            p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        }
        else {
            p.x += velocity * cos(p.theta) * delta_t;
            p.y += velocity * sin(p.theta) * delta_t;
        }
        p.theta += yaw_rate * delta_t;

        // Add noise
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto &p: particles) {
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        double cos_theta = cos(p.theta);
        double sin_theta = sin(p.theta);
        for (auto observation: observations) {
            double obs_x_p = p.x + observation.x * cos_theta - observation.y * sin_theta;
            double obs_y_p = p.y + observation.x * sin_theta + observation.y * cos_theta;
            int landmark_id = 0;
            double nearest_distance = numeric_limits<double>::max();

            // Find nearest landmark to observation
            for (auto &lm: predicted) {
                double distance = dist(obs_x_p, obs_y_p, lm.x, lm.y);
                if (distance < nearest_distance) {
                    landmark_id = lm.id;
                    nearest_distance = distance;
                }
            }
            sense_x.push_back(obs_x_p);
            sense_y.push_back(obs_y_p);
            associations.push_back(landmark_id);
        }
        p = SetAssociations(p, associations, sense_x, sense_y);
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double var_x = pow(std_x, 2);
    double var_y = pow(std_y, 2);

    // Particle with highest weight
    Particle& best = *max_element(particles.begin(), particles.end(), [](const Particle &a, const Particle &b) {
        return a.weight < b.weight;
    });

    vector<LandmarkObs> predicted;

    // Localize landmarks inside of 2 * sensor_range + noise
    double range = (sensor_range + std_x + std_y) * 2;
    for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
        LandmarkObs landmark;
        landmark.x = map_landmarks.landmark_list[i].x_f;
        landmark.y = map_landmarks.landmark_list[i].y_f;
        landmark.id = map_landmarks.landmark_list[i].id_i;
        if (dist(best.x, best.y, landmark.x, landmark.y) < range) {
            predicted.push_back(landmark);
        }
    }

    // Associate observations with landmarks

    dataAssociation(predicted, const_cast<vector<LandmarkObs> &>(observations));

    double total_weight = 0;
    for (int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        double log_weight = 0;
        for (int j = 0; j < p.associations.size(); j++) {
            double sense_x = p.sense_x[j];
            double sense_y = p.sense_y[j];
            double x_mean = map_landmarks.landmark_list[p.associations[j] - 1].x_f; // Associated landmark
            double y_mean = map_landmarks.landmark_list[p.associations[j] - 1].y_f;
            double xx = pow((sense_x - x_mean), 2) / var_x;
            double yy = pow((sense_y - y_mean), 2) / var_y;
            log_weight -= xx + yy;
        }
        p.weight = exp(log_weight);
        weights[i] = p.weight;
        total_weight += p.weight;
    }
    // Normalize weight
    for_each(weights.begin(), weights.end(), [total_weight](double &w) {
        w /= total_weight;
    });
    for_each(particles.begin(), particles.end(), [total_weight](Particle &p) {
        p.weight /= total_weight;
    });
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    random_device rd;
    default_random_engine gen(rd());
    std::vector<Particle> resampled_particles(num_particles);
    discrete_distribution<int> weights_sample(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; ++i) {
        resampled_particles[i] = particles[weights_sample(gen)];
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

	particle.associations = associations;
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
