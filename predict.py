import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse

def main():
    
    parser = argparse.ArgumentParser()  
    parser.add_argument('image_file', help='Image path')
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('--label_file', help='Class names file', default='label_map.json')
    parser.add_argument('--top_k', type=int, help='Top K pred', default=5)
    
    args = parser.parse_args()
        
    model = tf.keras.models.load_model(args.model_file,custom_objects={'KerasLayer':hub.KerasLayer})
    
    with open(args.label_file, 'r') as f:
        class_names = json.load(f)
        
    def process_image(image):
        image = tf.convert_to_tensor(image,tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = image/255
        image = image.numpy()
        return image

    def predict(image_path, model, top_k):
        im = Image.open(image_path)
        image = np.asarray(im)
        image = process_image(image)
        image = np.expand_dims(image, axis = 0)
        pred = model.predict(image)
        top_ps, top_k = tf.math.top_k(pred[0], top_k)
        top_ps = top_ps.numpy().tolist()
        top_k = top_k.numpy().tolist()
        top_ns = [class_names[str(value+1)] for value in top_k]
    
        return top_ps, top_k, top_ns

    top_ps, top_k, top_ns = predict(args.image_file, model, args.top_k)
    print(top_ps)
    print(top_k)
    print(top_ns)
    
if __name__ == "__main__":
    main()