import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn

from config.config import auto_encoder_training_intermediate_layer_size, auto_encoder_encoding_size, device
from encoder.abstract_encoder import AbstractEncoder
from encoder.auto_decoder import AutoDecoder
from encoder.auto_decoder_trainer import DecoderTrainer
from encoder.auto_encoder import AutoEncoder
from encoder.auto_encoder_trainer import loadTrainingData, EncoderTrainer
from encoder.one_hot_encoder import OneHotEncoder
from part import Part


class FullAutoEncoder(AbstractEncoder):
    preEncoder: AbstractEncoder
    encoder: AutoEncoder
    decoder: AutoDecoder

    def __init__(self, preEncoder: AbstractEncoder, encoder: AutoEncoder, decoder: AutoDecoder):
        self.preEncoder = preEncoder
        self.encoder = encoder
        self.decoder = decoder

    def get_encoding_size(self):
        return self.encoder.encode2.out_features

    def encode(self, part: Part) -> Tensor:
        return self.encoder.encodeTensor(self.preEncoder.encode(part).to(device))

    def decode(self, part: Tensor) -> Part:
        return self.preEncoder.decode(self.decoder.decodeTensor(part))

def loadPretrainedAutoEncoder() -> FullAutoEncoder:

    state = torch.load('../data/trained_auto_encoder_decoder.dat', map_location=torch.device(device))
    preEncoder = OneHotEncoder()

    encode1 = nn.Linear(preEncoder.get_encoding_size(), auto_encoder_training_intermediate_layer_size).to(device)
    encode2 = nn.Linear(auto_encoder_training_intermediate_layer_size, auto_encoder_encoding_size).to(device)
    decode1 = nn.Linear(auto_encoder_encoding_size, auto_encoder_training_intermediate_layer_size).to(device)
    decode2 = nn.Linear(auto_encoder_training_intermediate_layer_size, preEncoder.get_encoding_size()).to(device)

    encode1.load_state_dict(state['encode1'])
    encode2.load_state_dict(state['encode2'])
    decode1.load_state_dict(state['decode1'])
    decode2.load_state_dict(state['decode2'])

    encoder = AutoEncoder(encode1, encode2)
    decoder = AutoDecoder(decode1, decode2)
    return FullAutoEncoder(preEncoder, encoder, decoder)

def createAndSavePretrainedAutoEncoder():
    trainingData = loadTrainingData()
    trainer = EncoderTrainer(trainingData)
    preEncoder = OneHotEncoder()
    trainer.prepareTrainingData(preEncoder)

    encoderNetwork = trainer.trainAndPlotResults(batch_size=150, cycles=90, legend='Encoder', color='red')
    encoderNetwork.eval()
    encoder = AutoEncoder(encoderNetwork.fc1, encoderNetwork.fc2)

    trainer = DecoderTrainer(preEncoder, encoder)
    decoderNetwork = trainer.trainAndPlotResults(batch_size=150, cycles=160, legend='Decoder', color='green')

    state = {
        'encode1': encoderNetwork.fc1.state_dict(),
        'encode2': encoderNetwork.fc2.state_dict(),
        'decode1': decoderNetwork.fc1.state_dict(),
        'decode2': decoderNetwork.fc2.state_dict()
    }
    torch.save(state, '../data/trained_auto_encoder_decoder.dat')

    plt.title("Final Auto Encoder-Decoder Training")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0.0006, 0.0017)
    plt.show()

if __name__ == '__main__':
    #createAndSavePretrainedAutoEncoder()
    encoder = loadPretrainedAutoEncoder()