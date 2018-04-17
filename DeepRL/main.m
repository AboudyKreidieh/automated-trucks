%% Initialization

model = "simple_model";     % name of model
dt = 0.1;                   % simulation step size (s)
num_rollouts = 10;          % number of rollouts per iteration

load_system('simple_model')
set_param(strcat(model, "/dt"), "Value", string(dt))
set_param(model,'SignalLogging','off')


%% Simulation

for i=1:num_rollouts

    % ===== Reset =====

    % truck 1
    pos1_init = 20;      % position
    vel1_init = 0;      % speed
    accel1_init = 0;    % acceleration

    % truck 2
    pos2_init = 0;      % position
    vel2_init = 0;      % speed
    accel2_init = 0;    % acceleration

    % reinitialize all parameters
    set_param(strcat(model, "/veh1_pos"), "Value", string(pos1_init))
    set_param(strcat(model, "/veh1_vel"), "Value", string(vel1_init))
    set_param(strcat(model, "/veh1_accel"), "Value", string(accel1_init))
    set_param(strcat(model, "/veh2_pos"), "Value", string(pos2_init))
    set_param(strcat(model, "/veh2_vel"), "Value", string(vel2_init))
    set_param(strcat(model, "/veh2_accel"), "Value", string(accel2_init))


    % ===== Rollout =====

    sim(model)

end
