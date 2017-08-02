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
#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);;
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    //particles[num_particles]
    //Sample random points from the gaussian distributions
    num_particles = 1000;
    for(unsigned int i=0; i< num_particles ; i++)
    {
        Particle ptcl;
        ptcl.id = i;
        ptcl.x = dist_x(gen);
        ptcl.y = dist_y(gen);
        ptcl.theta = dist_theta(gen);
        ptcl.weight = 1;
        particles.push_back(ptcl);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine def;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    for(unsigned int i=0; i< num_particles ; i++)
    {
        //double curr_yaw_rate = dist_velocity(def);
        //double curr_velocity = dist_velocity(def);
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;
        //update the temporary state space using CTRV
        if(abs(yaw_rate) > 0.001)
        {
            x = x + (velocity/yaw_rate)*(sin(theta+yaw_rate*delta_t)-sin(theta));
            y = y + (velocity/yaw_rate)*(cos(theta)-cos(theta+yaw_rate*delta_t));
            theta = theta + yaw_rate *delta_t;
        }
        else{
            x = x + delta_t*(velocity)*cos(theta);
            y = y + delta_t*(velocity)*sin(theta);
            theta = theta;
        }
        //update the particles
        particles[i].x = x + dist_x(def);
        particles[i].y = y + dist_y(def);
        particles[i].theta = theta + dist_theta(def);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(unsigned int i = 0; i< observations.size(); i++){

        double min_dist = std::numeric_limits<double>::max();
        int nearest_id;
        for (unsigned j = 0; j < predicted.size(); j++){
            //dist.push_back(dist_id{dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y),observations[i].id});

            if(dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) < min_dist){
                min_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
                nearest_id = predicted[j].id;
            }
        }
        //assign the nearest id to the observation
        observations[i].id = nearest_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
    double weights_sum = 0;
    for( int i = 0; i< num_particles ;i++)
    {
        std::vector<LandmarkObs> pred_obs;

        double x = particles[i].x;
        double y = particles[i].y;

        double theta = particles[i].theta;
        for(unsigned j = 0 ; j < observations.size() ; j++){
            float obs_x = x + (observations[j].x*cos(theta) - observations[j].y*sin(theta));
            float obs_y = y + (observations[j].y*cos(theta) + observations[j].x*sin(theta));
            pred_obs.push_back(LandmarkObs{observations[j].id, obs_x,obs_y});
        }

        //Get the landmarks with the range of the particle
        std::vector<LandmarkObs> nearbylandmarks;
        for( int k = 0; k < map_landmarks.landmark_list.size() ; k++){
            float r = 0;
            double land_x = map_landmarks.landmark_list[k].x_f;
            double land_y = map_landmarks.landmark_list[k].y_f;

            r = sqrt((x-land_x)*(x-land_x)+(y-land_y)*(y-land_y));
            if(r < sensor_range){
                nearbylandmarks.push_back(LandmarkObs{map_landmarks.landmark_list[k].id_i,map_landmarks.landmark_list[k].x_f,map_landmarks.landmark_list[k].y_f});
            }
        }
        dataAssociation(nearbylandmarks, pred_obs);
        double mean_x,mean_y;
        //calculate the weights of the particles
        //distance of the observation from the actual observation
        for( int l=0; l<observations.size();l++){
            //grab landmark by its id
            double x = pred_obs[l].x;
            double y = pred_obs[l].y;
            for( int m=0;m<nearbylandmarks.size();m++){
                if(nearbylandmarks[m].id == pred_obs[l].id){
                    mean_x = nearbylandmarks[m].x;
                    mean_y = nearbylandmarks[m].y;
                    break;
                }
            }
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double temp_weight = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(x-mean_x,2)/(2*pow(std_x, 2)) + (pow(y-mean_y,2)/(2*pow(std_y, 2))) ) );
            particles[i].weight *= temp_weight;
        }
        weights_sum += particles[i].weight;
    }
    //normalize the weights to make sum of all weights equal to 1
    for(int i = 0; i < num_particles ; i++){
        particles[i].weight /= weights_sum;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//pick a random particle from the set of particles
    vector<Particle> resampled_particles;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, num_particles);

    //obtain the particle with largest weight
    double max_weight = particles[0].weight;
    for(int i=1;i<num_particles;i++){
        if(particles[i].weight>max_weight){
            max_weight = particles[i].weight;
        }
    }
    std::uniform_real_distribution<> dis_real(0, 2*max_weight);

    int random_index = dis(gen);
    double beta = 0;

    for(int i=0;i<num_particles;i++){
        beta = beta + dis_real(gen);
        double omega_index = particles[random_index].weight;
        while(omega_index < beta){
            beta = beta - omega_index;
            random_index = (random_index + 1)%num_particles;
            omega_index = particles[random_index].weight;
        }
        resampled_particles.push_back(particles[random_index]);
    }
    //assign the resamples particles to particles
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
