# ML_Face_Mask_detector
Trying to build a face mask detector based on machine learning using supervised learning

For the face mask detection we use MobileNetV2 which seems to be light weight and good to use for anylzing images in the event of putting it on a light system.
We used transfer learning to add layers and train it on our data.

For the face detection an already trained cascade classifier is used to detect the face to pass into our classifier. But issues arise when using it because it wasn't trained to detect masked faces. Thus sometimes the face is not well detected

Additionnaly the script to prepare the data to train a new haas cascade cassifier in openCV with mask pictures and unmasked pictures for the positive samples and a collection of empty rooms, street and landscapes for the negatives.

Sources for model choice: https://www.ijeat.org/wp-content/uploads/papers/v10i6/F30500810621.pdf
