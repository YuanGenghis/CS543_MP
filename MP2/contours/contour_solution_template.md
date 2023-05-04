## Contour Detection - Solution Template

**NOTE:** All values and figures in this template are examples that you will need to replace with your own results

1. **Method Description.** *TODO*: 
warm-up: As the easiest way to minimize these artifacts, after checking the convolve2d function description, we can simply add "boundary='symm'" to to that.
Smoothing: I used the existing gaussian_filter function from spicy library, the derivative should still be the same. after testing different sigma value in gaussian_filter, I think 2 would fit my function the best
Non-maximum Suppression: my thoughts is the nms algo goes through all the pixels on the gradient intensity matrix and finds the pixels with the maximum value in the edge directions. We seperate by pi/4 degree

2. **Precision Recall Plot.** *TODO*: Use [contour_plot.py](contours/../contour_plot.py) to add curves for the different methods that you implemented into a single plot.
   
   <div align="center">
      <img src="plot.png" width="60%">
   </div>

3. **Results Table.** *TODO*: Present the performance metrics for each implementation part on the val set in a table format

   | Method | overall max F-score | average max F-score | AP | Runtime (seconds) |
   | ----------- | --- | --- | ---  | --- |
   | Initial implementation | 0.404 | 0.460 | 0.315 | 0.006 |
   | Warm-up [remove boundary artifacts] | 0.424 | 0.484 | 0.359 | 0.016 |
   | Smoothing | 0.553 | 0.581 | 0.521 | 0.024 |
   | Non-maximum Suppression | 0.547 | 0.575 | 0.511 | 0.357 |
   | Val set numbers of best model [From gradescope] | 0.547 | 0.575 | 0.512 |

4. **Visualizations.** *TODO:* Include visualization on 3 images from val set showing the impact of your implementation for each part. Describe the effect of your implementation for each part on these images by pointing to relevant image regions. Comment on where does your code work well, where it doesn't and why? You may also add visualizations of your own images. Here is how you can lay out images in markdown.
   <div align="center">
      <img src="227092.jpg" width="30%" style="margin:10px;">
      <img src="227092-raw.png" width="30%" style="margin:10px;">
      <img src="227092-nms.png" width="30%" style="margin:10px;">
   </div>

5. **Bells and Whistles.** *TODO*: Include details of the bells and whistles that you
   tried here.

   *TODO*: Present the performance metrics for the bells and whistles in a table format
   
   | Method | overall max F-score | average max F-score | AP | Runtime (seconds) |
   | ----------- | --- | --- | ---  | --- |
   | Best base Implementation (from above) | | | | 
   | Bells and whistle (1) [extra credit]) | | | | 
   | Bells and whistle (2) [extra credit]) | | | |
   | Bells and whistle (n) [extra credit]) | | | |
