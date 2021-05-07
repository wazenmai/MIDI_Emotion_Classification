import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from src.model.net import SAN
from REMI.midi2event import analyzer, corpus, event

path_data_root = "./REMI/ailab17k_from-scratch_remi"
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
midi_dictionary = pickle.load(open(path_dictionary, "rb"))
event_to_int = midi_dictionary[0]

def main(args) -> None:
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.cuda:
        print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))

    if args.types == "AV":
        labels= ["Q1","Q2","Q3","Q4"]
        model = SAN( num_of_dim= 4, vocab_size= 339, lstm_hidden_dim= 128, embedding_size= 300, r=4)
        checkpoint_path = "./exp/PEmo/SAN_REMI_head4/av_remi_gpu1/epoch=96-val_loss=0.8408-val_acc=0.6715.ckpt"
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    elif args.types == "Arousal":
        labels= ["HA","LA"]
        checkpoint_path = "./exp/PEmo/SAN_REMI_2Class_head16/a_remi_gpu1/epoch=37-val_loss=0.3033-val_acc=0.8686.ckpt"
        model = SAN(num_of_dim=2, vocab_size= 339, lstm_hidden_dim= 128, embedding_size= 300, r=16)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    elif args.types == "Valence":
        labels= ["HV","LV"]
        checkpoint_path = "./exp/PEmo/SAN_REMI_2Class_head16/v_remi_gpu0/epoch=82-val_loss=0.5364-val_acc=0.7484.ckpt"
        model = SAN(num_of_dim=2, vocab_size= 339, lstm_hidden_dim= 128, embedding_size= 300, r=16)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()

    model = model.to(args.cuda)

    midi_obj = analyzer(args.midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    quantize_midi = [event_to_int[str(i['name'])+"_"+str(i['value'])] for i in event_sequence]

    torch_midi = torch.LongTensor(quantize_midi).unsqueeze(0)
    prediction = model(torch_midi.to(args.cuda))
    print("========")
    print(args.midi_path, " is emotion", labels[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()])
    print("Inference values: ",prediction)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="AV", type=str, choices=["AV","Arousal","Valence"])
    parser.add_argument("--midi_path", default="./dataset/sample_data/example_generative.mid", type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)
