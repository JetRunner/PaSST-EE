import h5py
import os


thres = 10000  # the files whose size < thres will be filtered out
hdf5_partition = "unbalanced_train_segments"

base_dir = "audioset_hdf5s/"
old_hdf5_file = os.path.join(base_dir, "mp3", f"{hdf5_partition}_mp3.hdf")
new_hdf5_file = os.path.join(base_dir, "mp3", f"new_{hdf5_partition}_mp3.hdf")

f = h5py.File(old_hdf5_file, "r")
print("Filtering", old_hdf5_file)

audio_names = f['audio_name']
mp3s = f['mp3']
targets = f['target']

valid_indices = []
for i in range(len(mp3s)):
    if len(mp3s[i]) > thres:
        valid_indices.append(i)
    if i % 1000==0:
            print(f"{i}/{len(mp3s)}")

print(valid_indices)
print("Save", len(valid_indices), "from", len(mp3s))

with h5py.File(new_hdf5_file, 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((len(valid_indices),)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((len(valid_indices),)), dtype=mp3s.dtype)
    target = hf.create_dataset('target', shape=((len(valid_indices), 66)), dtype=targets.dtype)
    for idx, i in enumerate(valid_indices):
        if idx % 1000==0:
            print(f"{idx}/{len(valid_indices)}")
        audio_name[idx]=audio_names[i]
        waveform[idx] = mp3s[i]
        target[idx] = targets[i]

print("Done!")