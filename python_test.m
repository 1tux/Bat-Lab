inp_behav_file='C:\Users\itayy\Documents\Bat Lab - saikat\data\behavioural_data\parsed\b2305_d191223_simplified_behaviour.csv';
inp_neural_file='C:\Users\itayy\Documents\Bat Lab - saikat\data\neural_data\parsed\82_b2305_d191223.csv';
config_file='C:\Users\itayy\Documents\Bat Lab - saikat\config.json';

% system('python main.py ');
execution_line = strcat('C:\Users\itayy\AppData\Local\Programs\Python\Python38-32\python.exe main.py "', inp_behav_file, '" "', inp_neural_file, '" "', config_file,'"');
disp(execution_line);
system(execution_line);
