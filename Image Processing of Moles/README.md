 ## Image Processing of Moles
 
Image of a mole is processed first with **k-means clustering** algorithm to reduce the number of colors to only four and then boundary of objects in new image is identified using ActiveContours MATLAB app.

### Medium risk mole

![image](https://user-images.githubusercontent.com/25234772/220735556-b96cf825-295b-4a65-92ad-31228b320f23.png)

Image obtained after using Active Contours after 500 iterations:
![image](https://user-images.githubusercontent.com/25234772/220735619-e0f8d416-5c5e-4571-a111-1382f56da018.png)

### High risk mole (Melanoma)

![image](https://user-images.githubusercontent.com/25234772/220735880-ee6d6f08-f1e3-4466-b00e-1a062ad748a1.png)

Image obtained after using Active Contours after 500 iterations:
![image](https://user-images.githubusercontent.com/25234772/220735931-ad20ffc9-e81f-4750-ae36-97df4a2483d7.png)

### Comments

Analyzing picture of a sample of mole from each category, it can be noted that as moles tend to grow from low risk to high risk, boundaries between different objects in image become more clear and distinct.

It can be concluded that it is easier to identify between different types of moles after reducing the number of colors and finding boundaries of contours in image. A very simple method like this could be of great use for analyzing and diagnosing images of moles in large numbers.
