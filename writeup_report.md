# Vehicle Detection Writeup

---

## HOG

I extracted HOG on all 3 channels of a YCrCb color space with 2 cells per block, 8 pixels per cell, and 7 orientations. The values for color space, pixels per cell, and orientations were reached through trial and error, and the cells per block was chose to be the smallest value that would give a good result in order to minimize the the size of the feature array. 

The code for the hog feature extraction is reproduced below:

```py
cells_per_block = (2,2)
pixels_per_cell = (8,8)
orientations= 7

image = cv2.resize(image, (64, 64))
	
ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
ch0 = ycrcb[:,:,0]
ch1 = ycrcb[:,:,1]
ch2 = ycrcb[:,:,2]
ch0_features = hog(ch0, orientations, pixels_per_cell, cells_per_block, transform_sqrt=gamma)
ch1_features = hog(ch1, orientations, pixels_per_cell, cells_per_block, transform_sqrt=gamma)
ch2_features = hog(ch2, orientations, pixels_per_cell, cells_per_block, transform_sqrt=gamma)
hog_features = np.concatenate((ch0_features, ch1_features, ch2_features))

```

---

## Other Features

In addition to HOG, color histograms for each channel and raw pixel values were used. The color histograms contained 32 bins for each channel. This was the default value given in the Udacity code snippet, and it worked well so I left it as is. For the raw pixel values, I first resized the image to 32x32 before flattening the pixel array. This was the smallest value where I could pick out the car features with the naked eye, so I assumed the algorithm would require that level of granularity as well. 

The code for the color histograms and pixel values is reproduced below:

```py
rhist, _ = np.histogram(image[:,:,0], bins=32, range=(0, 256))
ghist, _ = np.histogram(image[:,:,1], bins=32, range=(0, 256))
bhist, _ = np.histogram(image[:,:,2], bins=32, range=(0, 256))
color_hist_features = np.concatenate((rhist, ghist, bhist))

spatial_features = cv2.resize(image, (32, 32))

features = np.concatenate((hog_features.ravel(), color_hist_features.ravel(), spatial_features.ravel()))
```

---

## Classification

The LinearSVC implementation from scikit-learn was used to train against the input data. The input data was the GTI and KITTI datasets provided by Udacity for vehicles and non-vehicles. The features were normalized to zero mean and unit variance before training. 

The data was NOT shuffled before being split into training and validation sets. This is because the data contained a lot of very similar images in sequence, so shuffling the data would have essentially leaked signals about the validation set into the training set. Not shuffling the data provided a more generalizable implementation. 

The C value was tuned to prevent overfitting. The optimal C value where training and test accuracies were nearly equivalent at about 97% was 3e-7.

The code for the classifier is reproduced below.

```py
split_index = int(len(udacity_features_scaled) * .8)
X_train = udacity_features_scaled[0: split_index]
X_test = udacity_features_scaled[split_index:]
y_train = udacity_labels[0:split_index]
y_test = udacity_labels[split_index:]
# X_train, X_test, y_train, y_test = train_test_split(udacity_features_scaled, udacity_labels)

svc = LinearSVC(C=3e-7)
svc.fit(X_train, y_train)

train_accuracy = svc.score(X_train, y_train)
val_accuracy = svc.score(X_test, y_test)
```

---

## Sliding Windows

After the classifier was trained on the 64x64 images, a sliding windows approach was taken on each frame to identify regions with vehicles. Windows of size 115x115 and 90x90 were slid across the bottom right quarter of the input frame to identify vehicles. The classifier was extremely sensitive with the inclusion of spatial features, so in order to prevent false positives the window detections were passed through a thresholding operation and temporal high pass filter.

The thresholding operation produced a heatmap where each window detection for a given pixel raised the value of that pixel by 1 point. This heatmap was then passed through a high pass filter that gave 20% weight to the current frame and 80% weight to the historical heat map. After passing through the high pass filter, the heatmap is thresholded to filter out values less than 0.3 . A labeler then takes the thresholded values and groups them, each group representing one vehicle detection. A box approximation of tthese labeled zones is produced by taking the minimum and maximum x and y value for each label and then drawn on the original image. 

Here is the code for the sliding windows and detection:

```py
def pipeline(image, boxes=False):
    global heatmap
    draw_image = np.copy(image)
    search_spaces = [(400, 600, 115, 50), (400, 550, 90, 45)] 
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    local_heatmap = np.zeros((720, 1280))
    
    for color, space in zip(colors, search_spaces):
        windows = sliding_windows(*space)
        window_images = [image[y1:y2,x1:x2][np.newaxis, ...] for x1,y1,x2,y2 in windows]
        window_images = np.concatenate(window_images, axis=0)
        features = [feature_pipeline(window_image)[np.newaxis,...] for window_image in window_images]
        features = np.concatenate(features)
    
        features_scaled = scaler.transform(features)
        pred = svc.predict(features_scaled)   
        
        for window, guess in zip(windows, pred):
            if guess == 1:
				local_heatmap[window[1]:window[3], window[0]:window[2]] += 1
 
        
	heatmap = .8 * heatmap + .2 * local_heatmap
	thresholded = heatmap > .3
	car_labels = label_heatmap(thresholded)
	for i in range(car_labels[1]):
		indexes = (car_labels[0] == i+1).nonzero()
		upperLeft = (np.min(indexes[1]), np.min(indexes[0]))
		lowerRight = (np.max(indexes[1]), np.max(indexes[0]))
		cv2.rectangle(draw_image, upperLeft, lowerRight, (255, 0, 0), 5)
    
    return draw_image

def sliding_windows(ystart, ystop, size, step, shape=(720, 1280)):
    xstart = 750
    return [(x,y, x+size, y+size) for y in range(ystart, ystop - size, step) for x in range(xstart, shape[1] - size, step)]
```

An example output frame is shown below:

![Output Frame](assets/output_frame.png)

Additionally, here is an example of the sliding windows detecting vehicles before thresholding or the high pass filter

![Boxes](assets/boxes.png)

---

## Output Video

Here's a link to the [output video](processed_video.mp4)

The high pass filter and thresholding used to detect false positives is explained in the above section. 

---

## Discussion

Right now, in order to make the pipeline faster, only the bottom right of the image is being searched vehicles. This works because the car is in the left lane for the entirety of the project video, but those stop limits would have to be removed in order to detect vehicles to the left of the car. 

The entire pipeline depends on good lighting conditions. Already, shadows were causing a lot of false positives in initial detection that had to be filtered out. Poor lighting would likely exarcebate this problem and cause the pipeline to break down. 

The implementation is a bit slow right now, only able to process about 3 frames/sec on my (admittedly slow) laptop. Optimizations such as sub-sampling for the HOG portions , greater processing power, and likely a GPU would be required to speed up the detection to a point where it could operate in real time. 
