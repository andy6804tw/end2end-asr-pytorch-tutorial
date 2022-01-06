import torch 
from asr_functions import init_optimizer,init_transformer_model
import librosa
import torchaudio
import scipy.signal
import numpy as np


def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy().T
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound



    
def load_model():
    asr_checkpoint = torch.load("epoch_61.th")
    epoch = asr_checkpoint['epoch']
    metrics = asr_checkpoint['metrics']
    if 'args' in asr_checkpoint:
            args = asr_checkpoint['args']
    audio_conf = dict(sample_rate=args.sample_rate,
                            window_size=args.window_size,
                            window_stride=args.window_stride,
                            window=args.window,
                            noise_dir=args.noise_dir,
                            noise_prob=args.noise_prob,
                            noise_levels=(args.noise_min, args.noise_max))
    label2id = asr_checkpoint['label2id']
    id2label = asr_checkpoint['id2label']
    asr_model = init_transformer_model(args,label2id,id2label)

    asr_model.load_state_dict(asr_checkpoint['model_state_dict'])
    asr_model.eval()
    asr_model.cuda()
    # print(asr_model)
    return asr_model,audio_conf,label2id




def main():
    asr_model,audio_conf,label2id = load_model()

    
    audio = load_audio(path='4507-16021-0047.wav')
    
    #parameters of stft
    n_fft = int(audio_conf['sample_rate']*audio_conf['window_size'])
    win_length = n_fft
    window = scipy.signal.hann
    hop_length = int(audio_conf['sample_rate']*audio_conf['window_stride'])
    
    #STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # print(spect)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    
    #normalization
    mean = spect.mean()
    std = spect.std()
    spect.add_(-mean)
    spect.div_(std)
   
    spect = torch.unsqueeze(spect,0)
    spect = torch.unsqueeze(spect,0)
    spect = spect.cuda()
 
    input_length = torch.tensor([spect.size(3)])

    with torch.no_grad():
        output = asr_model.module.evaluate2(spect,input_length)
    print("predict:",output)
    
    

    
    

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python index.py