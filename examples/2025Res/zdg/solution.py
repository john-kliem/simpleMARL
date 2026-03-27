import numpy as np
import os
# from ray.rllib.policy.policy import Policy
import random
#Need an added import for competition submission?
#Post an issue to the github and we will work to get it added into the system!

#NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

#YOUR CODE HERE

#Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
	#Add Variables required for solution
	
    def __init__(self, env):
        pass

	#Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(self,agent_id:str, full_obs_normalized:dict, full_obs:dict, global_state:dict,info):
        #WARNING: If using global state you must ensure your entry can run on both RED and BLUE sides
        # State includes actual coordinate positions which are not the same on each side
        if full_obs[agent_id]['tagging_cooldown'] < 55:
            return self.attack_policy(full_obs,agent_id)
        
        random_policy = True
        for i in range(2):
            if full_obs[agent_id][('teammate_' + str(i), 'tagging_cooldown')] < 60:
                random_policy = False
                break
        
        if random_policy:
            if random.random() > .5:
                return self.attack_policy(full_obs,agent_id)
            else:
                return self.defence_policy(full_obs,agent_id)


        for i in range(3):
            if full_obs[agent_id][('opponent_' + str(i), 'has_flag')]:
                return (10,full_obs[agent_id][('opponent_' + str(i), 'bearing')])
        
        return self.defence_policy(full_obs,agent_id)

    def avoid_opponent(self, distance, on_side, threshold):
        in_range = distance < threshold
        return in_range and on_side

    def defence_policy(self,obs,agent_id):
        home_flag = np.multiply(obs[agent_id]['own_home_distance'],np.array([np.cos(np.deg2rad(obs[agent_id]['own_home_bearing'])), np.sin(np.deg2rad(obs[agent_id]['own_home_bearing']))]))

        avoid_vector = []
        for i in range(0):
            if obs[agent_id]["obstacle"+"_"+str(i)+"_distance"] < 20:
                obstacle_vector = np.array([np.cos(np.deg2rad(obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"])), np.sin(np.deg2rad(obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"]))])
                obstacle_vector = np.multiply(obs[agent_id]["obstacle"+"_"+str(i)+"_distance"],obstacle_vector)
                avoid_vector.append(obstacle_vector)
        if len(avoid_vector) > 0:
            x = 0
            y = 0
            for vector in avoid_vector:
                x = vector[0]
                y = vector[1]
            norm = np.linalg.norm(np.array([x,y]))
            avoid_vector = np.array([x/norm,y/norm])

        if obs[agent_id]['own_home_distance'] > 40:
            newHeading = home_flag
            if len(avoid_vector) > 0:
                newHeading = newHeading - avoid_vector
            newHeading = np.degrees(np.arctan2(newHeading[1],newHeading[0]))

            for i in range(0):
                error = obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"] - newHeading
                if obs[agent_id]["obstacle"+"_"+str(i)+"_distance"] < 10 and (error > -90 and error < 90):
                    newHeading = newHeading - 90 + error
            return (10,newHeading)
        
        target = 0
        target_distance = 0
        for i in range(3):
            if not obs[agent_id][('opponent_' + str(i), 'on_side')] and not obs[agent_id][('opponent_' + str(i), 'is_tagged')] and target == 0:
                target = obs[agent_id][('opponent_' + str(i), 'bearing')]
                target_distance = obs[agent_id][('opponent_' + str(i), 'distance')]
            elif not obs[agent_id][('opponent_' + str(i), 'on_side')] and not obs[agent_id][('opponent_' + str(i), 'is_tagged')] and obs[agent_id][('opponent_' + str(i), 'distance')] < target_distance:
                target = obs[agent_id][('opponent_' + str(i), 'bearing')]
                target_distance = obs[agent_id][('opponent_' + str(i), 'distance')]

        if target != 0:
            return (10,target)

        if obs[agent_id]['scrimmage_line_distance'] > 40:
            return (10,obs[agent_id]['scrimmage_line_bearing'])
        
        if obs[agent_id]['wall_2_distance'] > 40:
            return (10,obs[agent_id]['wall_2_bearing'])
        
        if obs[agent_id]['wall_0_distance'] > 40:
            return (10,obs[agent_id]['wall_0_bearing'])
        
        return (0,0)


    def attack_policy(self,obs,agent_id):
        captured_flag = obs[agent_id]['has_flag']
        opposing_flag = obs[agent_id]['opponent_home_bearing']
        half_line = obs[agent_id]['scrimmage_line_bearing']

        flag_vector = np.multiply(1, np.array([np.cos(np.deg2rad(opposing_flag)), np.sin(np.deg2rad(opposing_flag))]))

        half_line_vector = np.multiply(1, np.array([np.cos(np.deg2rad(half_line)), np.sin(np.deg2rad(half_line))]))

        avoid_vector = []

        for i in range(4):
            if obs[agent_id]["wall"+"_"+str(i)+"_distance"] < 10:
                wall_vector = np.array([np.cos(np.deg2rad(obs[agent_id]["wall"+"_"+str(i)+"_bearing"])), np.sin(np.deg2rad(obs[agent_id]["wall"+"_"+str(i)+"_bearing"]))])
                wall_vector = np.multiply(obs[agent_id]["wall"+"_"+str(i)+"_distance"],wall_vector)
                avoid_vector.append(wall_vector)

        for i in range(0):
            if obs[agent_id]["obstacle"+"_"+str(i)+"_distance"] < 20:
                obstacle_vector = np.array([np.cos(np.deg2rad(obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"])), np.sin(np.deg2rad(obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"]))])
                obstacle_vector = np.multiply(obs[agent_id]["obstacle"+"_"+str(i)+"_distance"],obstacle_vector)
                avoid_vector.append(obstacle_vector)

        for i in range(3):
            avoid = self.avoid_opponent(obs[agent_id][('opponent_' + str(i), 'distance')], obs[agent_id][('opponent_' + str(i), 'on_side')], 30)
            #print( 'opponent_' + str(i) + " " + str(obs[self.agent_id][('opponent_' + str(i), 'tagging_cooldown')]))
            if avoid :
                opponent_vector = np.array([np.cos(obs[agent_id][('opponent_' + str(i), 'bearing')]), np.sin(obs[agent_id][('opponent_' + str(i), 'bearing')])])
                co = np.divide(30,(obs[agent_id][('opponent_' + str(i), 'distance')]))
                avoid_vector.append([opponent_vector[0]*co,opponent_vector[1]*co])

        if len(avoid_vector) > 0:
            x = 0
            y = 0
            for vector in avoid_vector:
                x = vector[0]
                y = vector[1]
            norm = np.linalg.norm(np.array([x,y]))
            avoid_vector = np.array([x/norm,y/norm])

        if not captured_flag:
            newHeading = flag_vector
            if len(avoid_vector) > 0:
                newHeading = flag_vector - avoid_vector
            newHeading = np.degrees(np.arctan2(newHeading[1],newHeading[0]))
        else:
            newHeading = half_line_vector
            if len(avoid_vector) > 0:
                newHeading = half_line_vector - avoid_vector
            newHeading = np.degrees(np.arctan2(newHeading[1],newHeading[0]))

        for i in range(0):
            error = obs[agent_id]["obstacle"+"_"+str(i)+"_bearing"] - newHeading
            if obs[agent_id]["obstacle"+"_"+str(i)+"_distance"] < 10 and (error > -90 and error < 90):
                newHeading = newHeading - 75 + error
                    
        
        return (10,newHeading)
    
#END OF CODE SECTION
