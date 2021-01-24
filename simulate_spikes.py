import argparse
import sys
import spike_gen
import analysis_lib
import config
import os

def handle_args(args):
    parser = argparse.ArgumentParser(description='Simulate spikes & Analyze.')
    parser.add_argument('bpath', metavar='behavioral_path', type=str, nargs='?', help='Path for behavioral data (mat/csv)', default=r"C:\Users\itayy\Documents\saikat - Bat Lab\data\behavioural_data\parsed\b2305_d191223_simplified_behaviour.csv")
    parser.add_argument('-n', metavar='net', type=int, help='which net, could be 1 or 3', default=1)
    parser.add_argument('-X', metavar='eXclude', type=str, nargs='*', default=[])
    parser.add_argument('cpath', metavar='config_path', type=str, nargs='?', help='Path for configuration file (json)', default='config.json')
    args = parser.parse_args()

    config.Config.from_file(args.cpath)
    behavioral_data_path = args.bpath
    exclude = args.X

    try:
        net = {1: "NET1", 3: "NET3"}[args.n]
    except:
        raise Exception("Wrong Net! should have been either 1 or 3 %s" % str(args.n))

    return behavioral_data_path, net, exclude

    bat_name, day, _, _ = Path(behavioral_data_path).stem.split('_')

def main(behavioral_data_path, net, exclude):
    dataset = analysis_lib.behavioral_data_to_dataframe(behavioral_data_path, net, exclude)

    # neuron1 - place cell
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    print(simulated_spike.mean())
    simulated_spike &= spike_gen.gaussian_head_direction_cell(dataset, 80, 15)
    print(spike_gen.gaussian_head_direction_cell(dataset, 80, 15).mean())
    neuron1_path = r"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data\HD_place_cell.csv"
    simulated_spike.to_csv(neuron1_path)

    # neuron2 - OR place cell Pair23
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    neuron2_path = r"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data\OR_place_cell.csv"
    simulated_spike.to_csv(neuron2_path)
    
    # neuron3 - AND place cell Pair 23
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike &= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    neuron3_path = r"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data\AND_place_cell.csv"
    simulated_spike.to_csv(neuron3_path)

    os.system(f"python main.py \"{behavioral_data_path}\" \"{neuron1_path}\"")
    os.system(f"python main.py \"{behavioral_data_path}\" \"{neuron2_path}\"")
    os.system(f"python main.py \"{behavioral_data_path}\" \"{neuron3_path}\"")


if __name__ == "__main__":
    behavioral_data_path, net, exclude = handle_args(sys.argv)
    main(behavioral_data_path, net, exclude)