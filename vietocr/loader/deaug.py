import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

x = lambda a: a + a % 2 - 1
def sharpen(img):
  # ksize must be odd
  h = 3
  w = 3
  sharpener = - np.ones((h, w))
  sharpener[int(h / 2), int(w / 2)] = h * w
  sharpened = cv2.blur(img, (x(int(h/2)), x(int(w/2))))
  sharpened = cv2.filter2D(sharpened, -1, sharpener)
  sharpened = cv2.bilateralFilter(sharpened, h * w, 150, 75)
  return sharpened

def smooth(img, color_dis, scale=75):
  blurred = cv2.bilateralFilter(img, 21, color_dis, int(scale))
  sharpened = sharpen(blurred)
  return sharpened

# 1 cho chiều ngang
# 0 cho chiều dọc
def line_erosion(mask, axis = 0, threshold=0.9):
  dist = np.average(mask, axis=axis)
  print(dist.max(), dist.shape)
  for i in range(dist.shape[0]):
    if dist[i] >= threshold:
      # print("Deleted at i-th")
      if axis == 0:
        mask[:, i] = 0
      else:
        mask[i, :] = 0
  return mask

def sharp_mask(img, line_x_erosion, line_y_erosion):
  threshold = cv2.threshold(cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY), 200, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  print(threshold.shape)
  if np.average(threshold) > 0.75:
    threshold = 1 - threshold
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
  # plt.imshow(closed)
  # plt.pause(1)
  h, w = img.shape[:2]
  final = None
  if h > w:
    final = line_erosion(closed, 1, line_x_erosion)
    final = line_erosion(final, 0, line_y_erosion)
  else:
    print("Crop row first")
    final = line_erosion(closed, 0, line_y_erosion)
    final = line_erosion(final, 1, line_x_erosion)
  # final = cv2.morphologyEx(final, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
  return final

def siou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / boxA
	# return the intersection over union value
	return iou

def dropout_iou(boxes):
  i = 0
  while i  < len(boxes):
    for j in range(len(boxes)):
      if i != j:
        if siou(boxes[i], boxes[j]) > 0.6 and boxes[i][2] * boxes[i][3] < boxes[j][2] * boxes[j][3]:
          boxes.pop(i)
          i -= 1
      i += 1
  return boxes


def cropbbox(img, box):
  y, x, h, w = box
  return img[x:x+w, y:y+h].copy()

def dropout_axis(mask, axis = 0):
  dist = np.mean(mask, axis = 1 - axis)
  print(dist.shape)
  dist_avg = np.mean(dist)
  reduced = []
  location = []
  for i in range(mask.shape[axis]):
    if dist[i] > dist_avg and dist[i] < 0.75:
      location.append(i)
  loc = np.mean(location)
  size = len(location)
  return loc, size

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
def gen_checkpoint(mask, join=False):
  if join is True:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[len(cnts) % 2]

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  while len(cnts) > 50:
    mask = cv2.erode(mask, kernel)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[len(cnts) % 2]

  boxes = [cv2.boundingRect(cnt) for cnt in cnts]
  if len(boxes) == 0:
    return []

  cropped = [cropbbox(mask, box) for box in boxes]

  hm, wm = mask.shape[:2]
  ratio = np.sum(mask)
  select = []
  for i in range(len(cropped)):
    if np.sum(cropped[i]) > ratio / (len(cnts) * 1.75):
      convex = cv2.convexHull(cnts[i])
      area = cv2.contourArea(convex)
      x, y, w, h = boxes[i]
      if area > 0.4 * h * w and max(w, h) < max(wm, hm) * 0.75:
        locy, sizey = dropout_axis(cropped[i], axis = 0)
        select.append([x, y + max(0, h - w) / 2, w, h])
      else:
        print("Not fill enough")
  select = dropout_iou(select)
  print(select)
  return select

def word_angle(mask):
  select = gen_checkpoint(mask, False)
  if len(select) == 0:
    select = gen_checkpoint(mask, True)

  if len(select) == 0:
    return (0, 0), [[0]], True

  centers = [[box[0] + box[2]/ 2, box[1] + box[3]/ 2, box[2], box[3]] for box in select]
  centers_x = sorted(centers)
  print("Selecting {}".format(centers_x))
  occupied = np.zeros(mask.shape[1])
  reference = []
  for center in centers_x:
    if occupied[int(center[0])] == 0:
      reference.append(center[:2])
      begin = max(0, center[0] - center[2] / 2)
      end = min(occupied.shape[0], center[0] + center[2] / 2)
      occupied[int(begin) : int(end)] = 1
  if len(reference) < 2:
    return (0, 0), [[0]], True

  ref = np.array(reference)
  plt.imshow(mask)
  plt.scatter(ref[:, 0], ref[:, 1])
  plt.pause(0.5)

  A = np.stack([ref[:, 0], np.ones(ref.shape[0])], axis = 0).T
  param, residual = np.linalg.lstsq(A, ref[:, 1], rcond=None)[:2]

  return param, residual, True

def extract_median_color_axis(img, threshold, axis = 0):
    color_threshold = threshold

    img[img < color_threshold * 255] = 0
    mean_color = np.mean(img, axis=axis)
    mean_dist = np.mean(color_threshold, axis = axis)
    background = mean_color / (1 - mean_dist)
    return background

def reduce(mask, axis=1, median = True):
  distribution = np.average(mask, axis = axis)
  dist_avg = np.average(distribution)
  upper = 0.75
  if not median:
    dist_avg = 0
    upper = 2
  reduced = []
  for i in range(distribution.shape[0]):
    if distribution[i] > dist_avg and distribution[i] < upper:
      if axis == 1:
        reduced.append(mask[i, :])
      else:
        reduced.append(mask[:, i])
  if axis == 1:
    return np.vstack(reduced)
  else:
    return np.vstack(reduced).T

def seq_morph(mask, seq):
  for act in seq:
    mask = cv2.morphologyEx(mask, act[0], cv2.getStructuringElement(cv2.MORPH_RECT, (act[1], act[1])), iterations = act[2])
  return mask

def curve_func(x, a, b):
  return a * x + b

# since everything becomes very
def concentrate(mask, scale):
  norm = mask.copy()
  max_dropout = norm.max()
  norm[norm == max_dropout] = 0
  # plt.imshow(norm)
  # plt.pause(1)
  density = np.average(norm > 0)
  print(density)
  norm_avg = (np.average(norm) / np.average((norm > 0).astype(np.float32))) * scale
  # wash away noises
  norm[norm < norm_avg] = 0
  norm = np.interp(norm, (mask.max(), mask.min()), (0, 255))
  # plt.imshow(norm)
  return norm

def dist_angle(mask):
  # plt.imshow(mask)
  # plt.pause(1)
  box_size = np.array([mask.shape[0] / 5, mask.shape[1] / 5]).astype(int)
  # print("Blur size:{}".format(box_size))
  distribution = cv2.boxFilter(mask, -1, box_size)
  # plt.imshow(distribution)
  # plt.pause(1)
  pooled = max_pooling(distribution, axis = 0)
  scatter = concentrate(pooled, 0.8)
  # plt.imshow(scatter)
  # plt.pause(1)
  points = np.array(np.where(scatter > 0)).T
  if points.size == 0:
    print("Empty dataset !")
    return [0, 0], [], 0
  param, pcov = curve_fit(curve_func, points[:, 1], points[:, 0])
  # visualize(param, mask, 0)
  # plt.pause(1)
  angle = np.arctan(param[0])
  return param, pcov, angle * 180 / np.pi

def thinning(mask):
  #thinning word into a line
  # Structuring Element
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  # early stopping
  if cv2.countNonZero(cv2.erode(mask,kernel)) == 0:
    return mask

  # Create an empty output image to hold values
  thin = np.zeros(mask.shape,dtype='uint8')
  # Loop until erosion leads to an empty set
  while (cv2.countNonZero(mask)!= 0):
    # Erosion
    erode = cv2.erode(mask,kernel)
    # Opening on eroded image
    open = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
    # Subtract these two
    subset = erode - open
    # Union of all previous sets
    thin = cv2.bitwise_or(subset,thin)
    # thin = cv2.morphologyEx(thin, cv2.MORPH_DILATE, kernel, iterations=1)
    # thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, kernel, iterations=2)
    # thin = cv2.morphologyEx(thin, cv2.MORPH_ERODE, kernel, iterations=1)
    # Set the eroded image for next iteration
    mask = erode.copy()
  # thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, close_kernel, iterations=3)
  # thin = cv2.morphologyEx(thin, cv2.MORPH_DILATE, close_kernel, iterations=1)
  return thin

# chúng ta mặc định rằng các chữ sẽ có chiều dài > chiều rộng, bởi ngoại lệ chỉ có m
# ý tưởng cái này tựa tựa như Cluster của pad
def word_length(hw, angle):
  # với góc quay càng lớn trong khoảng (0, 90)
  # ta càng ít tự tin hơn việc cho rằng chiều dài của bbox là chiều dài của chữ hoặc từ
  # nên ta căn chỉnh cho dễ thở
  if hw[0] == 0 or hw[1] == 0:
    return 0, 0

  sigm = lambda a, angle: 1 / ( 1 + np.exp( a *(-np.tan(angle) + 1)))
  if hw[0] < hw[1] * 0.65:
    if angle < 30:
      return hw[0], angle
    # làm module nhạy cảm hơn với từ bị skew nhẹ
    elif angle > 75:
      if hw[1] / hw[0] >= 10:
        return 0, 0
      return hw[1] * 3, angle - 90
    else:
      a = sigm(4, angle)
      return hw[1] * (1 - a) + hw[0] * a, angle - 90
  else:
    # same shit
    if angle < 15:
      if hw[0] / hw[1] >= 10:
        return 0, 0
      return hw[0] * 3, angle
    elif angle < 45:
      a = sigm(4, angle)
      return hw[0] * a + hw[1] * (1 - a), angle
    elif angle > 60:
      a = sigm(4, angle + 10)
      return hw[0] * a + hw[1] * (1 - a), angle - 90
    elif angle >= 45:
      a = sigm(4, angle)
      return hw[0] * a + hw[1] * (1 - a), 90 - angle
  return 0, 0

# tính toán góc của từ dựa trên góc quay của các thành phần của nó :) căn chỉnh sao cho phù hợp
# đầu vào là đỉnh của từng thành phần
def rotate_angle(cnts, hw):
    h, w = hw
    area = h * w
    arects = [cv2.minAreaRect(cnt) for cnt in cnts]
    rects = []
    for rect in arects:
      if rect[1][0] * rect[1][1] > area / 100:
        rects.append(rect)

    angles = np.array([word_length(rect[1], rect[2]) for rect in rects])
    for rect in angles:
      print(rect)
    # print(angles)
    angle = np.sum(angles[:, 1] * angles[:, 0]) / np.sum(angles[:, 0])
    # print(angle)

    return len(rects), variance(angle, angles[:, 1], angles[:, 0]), angle

def variance(expected, obs, weights):
  return np.sum(np.abs(expected - obs) * weights) / np.sum(weights) 

def skew_angle(img, enhance=True, lowerb=0, export_mask = True):
  flag = False
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # đọc ảnh, trích xuất màu background
  bg_color = 0.
  threshold = None

  # line detection, we only delete in mask to prevent false recognition
  threshold = cv2.threshold(np.mean(img, axis=2).astype(np.uint8), 200, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  line_noise = False
  dist_x = np.average(threshold, axis = 1)
  for i in range(threshold.shape[0]):
    if dist_x[i] > 0.8:
      line_noise = True
      break

  if enhance:
    img = smooth(img, 20, 50)
    threshold = sharp_mask(img, 0.9, 0.9)
  else:
    threshold = cv2.threshold(np.mean(img, axis=2).astype(np.uint8), 200, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  plt.imshow(threshold)
  plt.pause(1)

  color_mask = img < (20, 20, 20)
  bg_color = extract_median_color_axis(img, color_mask, axis = (0, 1))
  bg_color = bg_color.astype(int).tolist()
  bg_color.append(0)
  print(bg_color)

  # lúc đút thì comment, chỗ này debug
  # plt.imshow(threshold)
  # plt.pause(1)


  # tìm đỉnh, xài đỉnh để detect hiệu quả k bị hao hụt quá so với xài edge mà chi phí rẻ
  cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[len(cnts) % 2]
  
  # prevent image to overflow
  # img_fk = img.copy()
  # for cnt in cnts:
  #   cv2.drawContours(img_fk, [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))],  0, (0, 255, 0), 2)

  min_rect = cv2.minAreaRect(np.concatenate(cnts))
  _, bbox_angle = word_length(min_rect[1], min_rect[2])
  bbox_dist = np.sum(threshold) / (min_rect[1][0] * min_rect[1][1])
  print("Bbox angle:{0} with confidence of {1}".format(bbox_angle, bbox_dist))

  num_box, var, char_angle = rotate_angle(cnts, min_rect[1])
  print(char_angle)
  h, w = img.shape[0:2]
  print("Character angle:{0} with variance of {1}".format(char_angle, var))

  # print("Min rect angle:{}".format(min_rect[2]))
  box_size = np.max(min_rect[1])
  _, _, dst_angle = dist_angle(threshold.astype(np.float32))
  print("Distribution angle: {}".format(dst_angle))


  param, loss, chain_success = word_angle(threshold.copy())
  chain_angle = np.arctan(param[0]) * 180 / np.pi
  print("Chaining angle {0} is {1}".format(chain_angle, chain_success))

  final= 0.

  # xoay lệch hướng có thẻ khiến cho ảnh bị sai nặng hơn :)
  # và tôi tin tưởng kết quả của bản thân hơn cái cv2 này
  # # tính xem góc từ hàm có bị bias cho chữ thay cho dòng chữ không

  reduced = reduce(threshold, axis = 1)

  reduced = seq_morph(reduced, [[4, 5, 1]])
  reduced = reduce(reduced, axis = 0, median = False)
  print("Text density:{}".format(np.average(reduced)))

  if np.average(reduced) > 0.8 or num_box <= 2:
    print("Text too filled or insufficient data for parsing")
    final = 0
    flag = True
    return bg_color, flag, final
  

  if num_box >= 5:
    if abs(char_angle) < lowerb and var <= 5:
      print("Preserve original ")
      final = 0
      flag = True
      return bg_color, flag, final
  

  if chain_success is True:
    print((chain_angle - bbox_angle) / (chain_angle - dst_angle))
    if (chain_angle - bbox_angle) / (chain_angle - dst_angle) < 0:
      final = chain_angle
    elif bbox_dist > 0.5:
      final = bbox_angle
    else:
      final = dst_angle
  else:
    if abs(dst_angle - char_angle) * abs(bbox_angle - char_angle) < 0:
      final = dst_angle
    elif abs(bbox_angle - char_angle) < abs(bbox_angle - char_angle) or bbox_dist > 0.5:
      final = bbox_angle
    else:
      final = dst_angle


  print("Final angle:{}".format(final))

  if abs(final) < lowerb:
    print("Preserve original")
    flag = True
  if not export_mask:
    return bg_color, flag, final
  else:
    return threshold, bg_color, flag, final
  

def max_pooling(mask, axis):
  h, w = mask.shape[:2]
  if axis > 1 | axis < 0:
    return mask
  metric = np.max(mask, axis = axis)
  if axis == 0:
    for i in range(w):
      z = mask[:, i]
      z[z < metric[i]] = 0
  elif axis ==1:
    for i in range(h):
      z = mask[i, :]
      z[z < metric[i]] = 0
  return mask




def dropout_angle(mask, angle):
  h, w = mask.shape[:2]
  x = np.arange(-w/2, w/2 )
  y = np.arange(-h/2, h/2 ).T
  X, Y = np.meshgrid(x, y)
  Y[Y == 0.] = 0.1
  dropout = np.abs(X / Y)

  ratio = 1 / np.tan(angle * np.pi / 180)

  mask[dropout < ratio] = 0

  return mask

def slant_correction(img, mask, bg_color):
  print(img.shape, mask.shape)
  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[len(cnts) % 2]
  min_rect = cv2.minAreaRect(np.concatenate(cnts))
  _, angle = word_length(min_rect[1], min_rect[2])
  skew_angle = angle * 3.14 / 180
  print("Skew angle:{}".format(angle))
  # plt.imshow(mask)
  # plt.pause(1)
  char_dft = np.fft.fft2(mask)
  dft_shift = np.fft.fftshift(char_dft)

  dft_mask =  20 * np.abs(dft_shift)
  dft_mask = dropout_angle(dft_mask, 40)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


  dft_norm = np.interp(dft_mask,
                     (dft_mask.min(), dft_mask.max ()),
                     (0, 255)).astype(int).astype(np.uint8)

  dft_norm = max_pooling(dft_norm, axis = 0)
  # bỏ đi điểm tâm cũng k ảnh hưởng bởi nó đằng nào cũng đối xứng qua tâm
  dft_norm[dft_norm > 250] = 0

  dft_avg = np.average(dft_norm) * img.shape[0]
  # print("Average:{}".format(dft_avg))
  dft_norm[dft_norm < dft_avg] = 0

  points = np.array(np.where(dft_norm > 0)).astype(int).T
  print("Points:{}".format(points.shape))

  A = [[point[1], 1] for point in points.astype(float)]
  a, b = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]
  print(a, b)
  view = dft_norm.copy()
  bx = np.array([0, 100])
  by = (bx * a + b).astype(int)
  cv2.line(view, (bx[0], by[0]), (bx[1], by[1]), color=255)
  # plt.imshow(view)
  # plt.pause(1)

  # # refine the shear based on "skew" angle
  shear_angle = np.arctan(a)
  # # print("Skew angle:{}".format(skew_angle))
  print("Shear angle:{}".format(shear_angle))
  shear_kernel = np.float32([[1, a, - a * (img.shape[0] - 5) / 2 ],
                         [0, 1, 0],
                         [0, 0, 1]])

  print(img.shape, mask.shape)
  masked_img = np.concatenate([img, np.expand_dims(mask, axis=2)], axis = 2)
  desheared_img = cv2.warpPerspective(masked_img, shear_kernel, dsize = [img.shape[1], img.shape[0]], borderMode=cv2.BORDER_CONSTANT, borderValue = bg_color,flags=cv2.INTER_LINEAR)
  # plt.imshow(desheared_img, 'gray')
  return desheared_img[:, :, :3], desheared_img[:, :, 3]

def rotate_img(img, angle, mask, bg_color):
  h, w = img.shape[:2]
  rotate_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.)
  masked_img = np.concatenate([img, mask[:, :, None].astype(np.uint8)], axis=2)
  rotate_matrix[:, 2] += [w / 8, h/ 8]
  iscale = 0.8
  output_size = (int(w / iscale), int(h / iscale))
  warped = cv2.warpAffine(src=masked_img,
                          M=rotate_matrix,
                          dsize=output_size,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)
  return warped[:, :, :img.shape[2]], warped[:, :, -1]

def crop_img(img, mask, pad_bx, pad_by, pad_ex, pad_ey, bg_color, smoothing=False):
  if smoothing:
    img = smooth(img, 20)
  
  points = np.array(np.where(mask > 0)).T
  y, x, h, w = cv2.boundingRect(points)
  bx = int(x - pad_bx)
  by = int(y - pad_by)
  ex = int(x + w + pad_ex)
  ey = int(y + h + pad_ey)
  ex_pad = int(max(ex - img.shape[1], 0))
  ey_pad = int(max(ey - img.shape[0], 0))
  bx_pad = - min(bx, 0)
  by_pad = - min(by, 0)
  print(bx, ex, bx_pad, ex_pad)
  print(by, ey, by_pad, ey_pad)
  cropped = img[max(0, by) : min(img.shape[0], ey), max(0, bx) : min(img.shape[1], ex)]
  padded = cv2.copyMakeBorder(cropped,
                              by_pad, ey_pad,
                              bx_pad, ex_pad,
                              borderType=cv2.BORDER_CONSTANT,
                              value=bg_color)
  return padded

def crop_original(img, pad_bx, pad_by, pad_ex, pad_ey):
  sub_img = smooth(img, 20, 75)
  mask = sharp_mask(sub_img, 0.85, 0.85)

  # plt.imshow(mask)
  # plt.pause(0.2)

  bg_mask = sub_img < (20, 20, 20)
  bg_color = extract_median_color_axis(img.copy(), bg_mask, axis = (0, 1)).astype(np.uint8).tolist()
  points = np.array(np.where(mask > 0)).T
  y, x, h, w = cv2.boundingRect(points)
  bx = int(x - pad_bx)
  by = int(y - pad_by)
  ex = int(x + w + pad_ex)
  ey = int(y + h + pad_ey)
  ex_pad = int(max(ex - img.shape[1], 0))
  ey_pad = int(max(ey - img.shape[0], 0))
  bx_pad = - min(bx, 0)
  by_pad = - min(by, 0)
  print(bx, ex, bx_pad, ex_pad)
  print(by, ey, by_pad, ey_pad)
  cropped = img[max(0, by) : min(img.shape[0], ey), max(0, bx) : min(img.shape[1], ex)]
  padded = cv2.copyMakeBorder(cropped,
                              by_pad, ey_pad,
                              bx_pad, ex_pad,
                              borderType=cv2.BORDER_CONSTANT,
                              value=bg_color)
  return padded

def process_img(img, skew=True, slant=True, smoothing=True):
  if skew is True:
    mask, bg_color, singular, angle = skew_angle(img, enhance=True, lowerb=10)
    rotated, rotated_mask = rotate_img(img, -angle, mask, bg_color)
    if slant is True:
      final, final_mask = slant_correction(rotated, rotated_mask, bg_color)
    output = crop_img(final, final_mask, 3, 3, 0, 1, bg_color[:3], smoothing)
    return output
  else:
    return crop_original(img, 2, 5, 3, 1)
