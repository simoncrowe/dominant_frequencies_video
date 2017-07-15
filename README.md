## dominant_frequencies_video
Provides methods for getting dominant pixel hue and audio frequency from a given frame of a video file.
## To use
- Clone the repository to where the script you want to use it in is located. 

i.e. git clone https://github.com/simoncrowe/dominant_frequencies_video

- Import it into your script. 

i.e. from dominant_frequencies_video.av_frequencies import VideoFileFreqAnalyser

- Create a VideoFileFreqAnalyser instance with your audio and video files. 

e.g. analyser =  VideoFileFreqAnalyser('video.mp4', 'audio.wav') *

- Use the get_dominant_hue(frame_number, sample_spacing = 32, hue_granularity = 256, segments= (1, 1)) function to get hue values 

e.g. analyser.get_dominant_hue(900)

- Use the get_dominant_frequency(frame_number) function to get the frequency of the samples for a particular frame in Hertz.

* _Due to efficiency issues with moviepy's handling of audio, the audio must be loaded as a separate WAV file. This file should be a single-channel WAV, ideally using 16 bit signed integers and can be created using ffmpeg_

e.g. ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 audio.wav

 ## Other possibe uses

While this script has quite a limited use case, some of its functionality could be re-used. The main functions of interest are:
- Getting the dominant frequency in Hertz from an array of audio samples using the fast Fourier transform from the numpy library. The most useful thing here is that the FFT data is generated and other analyses could be carried out on it.
- Generating an array of audio samples (16bit signed integers) containing a sine wave of a particular duration and frequency. Admittedly this function is quite trivial, but reusable nevertheless.
- Least useful probably: a set of functions of my own invention of getting the dominant hue band of a frame of video as a normalised float. Not guaranteed to be optimal or bug-free.
