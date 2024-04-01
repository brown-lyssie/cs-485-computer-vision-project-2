import project2 as p2

img1 = p2.load_img("cat.jpg")
# keypoints = p2.moravec_detector(img1);
# plotted_img1 = p2.plot_keypoints(img1, keypoints)
# p2.display_img(plotted_img1)
print(p2.calc_lbp_for_window([[5,2,1],[1,3,4],[2,3,1]]))
print(p2.extract_LBP(img1, [50, 70]))