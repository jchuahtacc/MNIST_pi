
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prod = tf.matmul(x, W)
y = tf.nn.softmax(prod + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import threading
import io
import time

stream = BytesIO()
camera = PiCamera()

width, height = camera.resolution
capture_size = 28 * 12 * 2
capture_x = width / 2 - capture_size / 2
capture_y = height / 2 - capture_size / 2
downsample_preview_scale = 8

capture_threshold = 100

status_font = ImageFont.truetype("/usr/share/fonts/truetype/roboto/Roboto-Light.ttf", 40)

capture_zone = Image.new('RGB', (capture_size, capture_size), color=(255, 255, 255))
preview_image = Image.new('RGB', (width, height), color=(128,128,128))
preview_image.paste(capture_zone, box=(capture_x, capture_y))

preview_draw = ImageDraw.Draw(preview_image)

camera.start_preview()
camera.preview.alpha = 240
overlay = camera.add_overlay(preview_image.tobytes(), layer=3, alpha=178)

def matrix_to_image(t):
	tnp = np.asarray(t)
	tnp = np.log(tnp + 2) * 255 - (176-128)
	tnp = tnp.astype(int)
	t_image = Image.fromarray(np.transpose(tnp)).resize((len(t) * 2, len(t[0]) * 10))
	return t_image

def vector_to_image(v):
	vnp = np.asarray(v)
	vnp = np.interp(vnp, np.linspace(np.min(vnp), np.max(vnp), 255), np.linspace(0, 255, 255)).astype(int)
	vnp = vnp.reshape(1, len(vnp))
	t_image = Image.fromarray( np.transpose(vnp) ).resize((8, len(vnp[0]) * 10))
	return t_image

	

w_left = 28 * (downsample_preview_scale + 1)
w_upper = 16
b_left = w_left + 784 * 2 + 32
prod_left = b_left - 16


def status(text):
	global status_font
	global preview_draw
	preview_draw.rectangle([(1350, 520), (1850, 620)],fill=(0,0,0,0))
	preview_draw.text((1400, 540), text, font=status_font)

print "Training..."
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	_, Wextract, bextract = sess.run([train_step, W, b], feed_dict={x: batch_xs, y_: batch_ys})
	if (i < 50) or (i % 50 == 0):
		W_image = matrix_to_image(Wextract)
		preview_image.paste(W_image, box=(w_left, w_upper))

		b_image = vector_to_image(bextract)
		preview_image.paste(b_image, box=(b_left, w_upper))
		status("Training: " + str(i / 10.0) + "%")

		overlay.update(preview_image.tobytes())
		time.sleep(0.1)


print "Training complete"

for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
	stream.seek(0)
	image = Image.open(stream)
	image = image.convert(mode='L')
	image = image.crop(box=(capture_x, capture_y, capture_x + capture_size, capture_y + capture_size))
	image = image.resize((28, 28), Image.ANTIALIAS)

	xs = np.array(image).reshape(1, 784)
	black_indexes = xs < capture_threshold
	xs[black_indexes] = 0
	white_indexes = xs >= capture_threshold
	xs[white_indexes] = 255
	xs = (255 - xs)
	image = Image.fromarray(np.resize(xs, (28, 28)))

	xs_percent = xs / 255.0
	
	y_extract, prod_extract = sess.run([y, prod], feed_dict={x: xs_percent })
	prediction = sess.run(y, feed_dict = { x: xs_percent }).argmax()
	status("I see the number " + str(prediction))
	prod_image = vector_to_image(prod_extract[0])
	preview_image.paste(prod_image, box=(prod_left, w_upper))
	y_image = vector_to_image(y_extract[0])
	preview_image.paste(y_image, box=(b_left + 16, w_upper))

	image = image.resize((28 * downsample_preview_scale, 28 * downsample_preview_scale), Image.ANTIALIAS)
	image_vector = Image.fromarray(xs, mode='L')
	image_vector = image_vector.resize((784 * 2, 8))
	#image_vector = image.resize((784 * 2, 8))
	preview_image.paste(image, box=(0, 0))
	preview_image.paste(image_vector, box=(28 *  (downsample_preview_scale + 1), 0))
	overlay.update(preview_image.tobytes())

	stream.seek(0)
	stream.truncate()

camera.remove_overlay(overlay)
camera.stop_preview()
