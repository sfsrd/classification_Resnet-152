import splitfolders

input_folder = 'path to folder with images'
output = 'path where to save'
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1))