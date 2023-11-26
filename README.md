# Mandelbrot Set Image Generator
![Full Mandelbrot Set](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/grayscale/image.png?raw=true)
## Project Description
This project is implementing a Mandelbrot Set Image Generator. The Mandelbrot Set is a escape time fractal and are all the points in the complex plane $c$ such that $f_c(z) = z^2 + c$ doesn't diverge when iterated from $z = 0$ i.e the point $c$ for which the sequence $f_c(0), f_c(f_c(0)), f_c(\ldots f_c(0) \ldots)$ remains bounded by absolute value. All the values inside the set are colored black or white and all the points outside the set are colored depending on some rule. While it is impossible to check for an infinite number of iterations, if you check for a suitably large amount of iterations you can create a approximation of the Mandelbrot set. Moreover 2 is typically used as the escape radius because if any value in the sequence has a absolute value greater than or equal to 2 the sequence is guaranteed to converge. The process of generating an approximation to the Mandelbrot set is extremely computationally expensive as you have to test each pixel for a suitably large number of iterations, but there are various optimizations you can make to simplify the calculation as well as prevent yourself from having to compute too many values. Most optimizations try to prevent the computation of values inside the mandelbrot set as those serve as the bottleneck of such a computational program as they are the most computational expensive. The optimizations include period checking, bulb checking, avoidance of the square root and complex number, use of quadtrees, and exploitation of symmetry.

## Table of Contents

- [Installation](##Installation)
- [Usage](##Usage)
- [Documentation](##Documentation)
- [References](##References)

## Installation

    git clone https://github.com/your-username/your-project.git
    cd your-project
    pip install -r requirements.txt

## Usage
Once you have installed the project properly and installed the necessary packages, to run the project you run:

    python main.py <directory>

Where directory is the directory where you want the image to be created. If a directory isn't provided, it will create a directory for you with default parameters. The project uses a configuration file system to provide parameters to run the project. The parameters include:

- ``accelerated`` - Whether to use the normal mandelbrot generator or the accelerated mandelbrot generator
- `height` - Height of the image
- `width` - Width of the image
- `center_real` - Real value of the complex number corresponding to center of image
 - `center_imaginary` - Real value of the complex number corresponding to center of image
- `world_width` - Difference between the real values of the left most and right most value of the screen. Determines in world width of viewport
- `max_iterations`: Determines the maximum iterations needed before determining whether a point is inside the mandelbrot set or not
- `escape_radius` : Determines $r_{escape}$ which is the aboslute value before which a point is guaranteed to diverge
- `color_scheme`: Color scheme used to color the image
- `builtin_colormap`: Uses built in matplotlib colormap or not
- `color_map`: If `color_scheme = 4 and builtin_colormap`, name of matplotlib colormap used to color, else if `color_scheme = 4` user provided colormap. The user provided colormap is a file inside the directory where each line are 3 integers from between 0 and 255 which are seperated by spaces corresponding to a RGB color.
- `smooth`: Is smooth coloring used
- `raster`: is rasterization used to generate
- `mixed_raster`: is mixed quadtree raster algorithm used to generate
- `fast_quadtree`:if accelerated mode is used should the fast quadtree implementations of the algorithms used
- `show_quadtree`: overlay quadtrees over image of mandelbrot set to visualize the quadtree datastructure

If there is a configuration file within your directory called `config.yml` the main.py script will use it to generate the image, if it isn't provided it will use the default parameters. The user's `config.yml` file must be in valid yaml configuration. Furthermore only the parameters which are different from the default configuration need to be provided.

The default configuration is as follows:

```yaml
---
accelerated: true
height: 512
width: 512
center_real: -0.75
center_imaginary: 0.0
world_width: 3.5
max_iterations: 100
escape_radius: 2.0
color_scheme: 0
colormap: None
smooth: false
raster: false
mixed_raster: false
fast_quadtree: true
show_quadtree: false
...
```
Example configuration files with their results are included in the examples directory. To calculate the viewport the following calculation is used.

```python
scale = world_width/width
offset = center + np.array([-width, height]) / 2
self.x = np.linspace(0, size[0], num=size[0], dtype=np.float64) * scale + offset[0]
self.y = np.linspace(0, size[1], num=size[1], dtype=np.float64) * -scale + offset[1]
```


## Implementation
Most optimizations of the Mandelbrot set seek to decrease the computational cost of finding points within the Mandelbrot set as those serve as the bottleneck of the program. The first few optimizations seek to modify the calculation function which determines whether a point is or isn't in the mandelbrot set.
1. **Avoidance of Square Roots**: When we calculate the absolute value of a complex number we have to take a square root because for all $z \in \mathbb{C}$, $|z| = \sqrt{\Re(z)^2  + \Im(z)^2}$. Taking the square root of a number is reasonably computationally inefficient compared to squaring a number so avoiding doing repeated square roots saves time as a whole. So instead of comparing $|z| \geq r_{escape}$, comparing $\Re(z)^2  + \Im(z)^2 \geq r_{escape}^2$ is much more efficient.
2. **Bulb/Cardioid Checking**: There is a known formula of the main body of the mandelbrot set so just checking if the coordinates are within the main body with this formula prevents the useless computation of the largest section of the set
3. **Exploitation of Symmetry**: The mandelbrot set is symmetric about the real axis. So a dictionary is used to save the values of the set which have already been computed so recomputation is not necessary additionally prior to any computation the absolute value of the imaginary component of the coordinates is taken.
4. **Period Checking**: To avoid having to see the sequence for the maximum number of iterations, you can break out of the computational look if you see that there is a repetition of values in the sequence as it would then produce a periodic pattern. Periods of up to 20 are checked which is a significantly large period as there are $2^{19}$ structures (closed subsets) within the Mandelbrot set which have periods of 20.

While these optimizations do make quite a large difference in the runtime of the generator, one of the largest optimizations comes through the use of quadtrees. The most computationally expensive part of calculating whether things are in the mandelbrot set or if they aren’t lies in the determination of values which are in the set. However the mandelbrot set has an interesting behavior that for any closed curve which is completely inside the set, everything inside it is also in the set. A quadtree is a tree data structure where any node may have up to 4 children depending on some criteria. Using the data structure helps us pinpoint the areas of our image which we care about (depending on the algorithm). So to exploit this, the domain is split into quadrants using some algorithms enumerated below. In some areas of the complex plane the use of Quadtree's causes a massive jump in performance.

Before talking about the algorithms it is important to talk about the coloring choices available as a particular algoirthms performance is directly tied to the coloring algorithm used as well as the area of the complex plane we have chosen to look at.

For each coloring scheme, there are two types of colorings available:
- **Discrete Coloring**: The discrete coloring of the set uses the raw iteration value outputted by the calculation function which calculates the number of iterations needed for the sequence to diverge (max_iteration if sequence doesn’t diverge within the maximum iteration) and then applies the coloring scheme to this value.

-  **Smooth Coloring**: Smooth colorings of the set applies a function to iteration value by taking in the final value of the sequence, and the escape radius used. Let x be the number of iterations needed to diverge and let z be the final value, dz be the derivative, and let r be the escape radius. Let $X=x- log(log(|z|2)) +log(log(r_{escape})) + 1$.  X is the continuous iteration of the point. Instead of using x in the color scheme it substitutes X. For smoother looks, closer to the set a higher value of $r_{escape}$ would be preferable.

The color schemes allowed are the following:
- Grayscale: Uses the iteration or continuous iteration divided by the maximum number of iterations to find grayscale value
- Simple HSV: Uses the iteration or continuous iteration divided by the maximum number of iterations to find hue value for HSV
-  Quilt Coloring: Uses the iteration or continuous iteration, final point before divergence, to create a quilt like coloring.
- Colormap: Either using the Matplotlib built in colormaps or a user created color map the set is colored

There are 3 algorithms which are used to render the Mandelbrot set each with their strengths and weaknesses.
- **Raster**: Basic rasterization goes through all of the pixels and calculates all of them. It is able to use all of the optimization techniques with the exception of quadtrees.
	- Accelerated implementation: The rastering process is parallelized by row using numba.
- **Mixed Raster and Quadtree**: The purpose of the algorithm is to use the quadtree to find the pixels which correspond to points in the mandelbrot set, which would be the most difficult aspect of the computation, and rasterize the rest which would be the relatively easier part of the computation. A quadtree is initialized with bounds covering the entire image. It is split into 4 child quadtrees which each correspond to a quadrant of the image. All these children are added to a queue. Breath First Search is used to traverse the tree level by level. A quadtree is popped from the queue and all of the colors of its boundary pixels are computed and stored. If all of the boundary pixels are inside the mandelbrot set it flood fills everything inside the section of the pixel array which the quadtree corresponds to. If some of the boundary pixels are in the mandelbrot set and others aren’t, then it splits the quadtree into 4 children while ignoring the boundary. For example, it wants to split a quadtree with dimensions 16 x 16, it ignores the boundary so considers the square which is 14 x 14 and then splits this square into quadrants so now each of the smaller quadtrees have dimensions 7 x 7. 1.  This is done to prevent unnecessary recomputation. If none of the boundary pixels are in the mandelbrot set, the quadtree is left as is. After the quadtree is split its children are pushed into the queue to be computed in the future. After BFS has finished and the queue is empty with the computation of the values inside of the mandelbrot set, then rasterization kicks in and calculates the remaining points which hadn’t been computed.
	- Accelerated Implementation: The rastering process is parallelized. In the fast_quadtree version of the quadtree computation. Instead of computing 1 quadtree at a time, it takes all the quadtree's in the queue at once and computes whether it needs to fill or split at once using parallelization. So it computes things in batches.
- **Normal Quadtree**: The algorithm is a more pure form of the algorithm above. A quadtree is initialized to the entire image. First the quadtree is split into 4 child nodes corresponding to its quadrants. A queue is created and the children of the first quadree are inserted. Then breath first search is used to traverse the quadtree. Where a quadtree is popped from the queue and all of the colors of its boundary pixels are computed and stored. If all of the boundary pixels have the same color it floods everything inside the section of the pixel array which the quadtree corresponds to. If the boundary pixels don’t have the same colors, it splits the quadtree into 4 children while ignoring the boundary. For example, it wants to split a quadtree with dimensions 16 x 16, it ignores the boundary so considers the square which is 14 x 14 and then splits this square into quadrants so now each of the smaller quadtrees have dimensions 7 x 7. If none of the boundary pixels are in the mandelbrot set, the quadtree is left as is. After the quadtree is split its children are pushed into the queue to be computed in the future.
	- 	Accelerated Implementation: Instead of computing 1 quadtree at a time, it takes all the quadtree's in the queue at once and computes whether it needs to fill or split at once using parallelization. So it computes things in batches.

![enter image description here](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/quadtree_view/image.png?raw=true)

Coloring and the choice of domain is what differentiates the performance of the algorithms. The Normal Quadtree algorithms perform the best when the Mandelbrot Set performs the best when a large portion of the points within the screen are within the set and when a discrete coloring algorithm is used. This is because it effectively splits the pixel area of blocks of the same color (which is easy since the choice of colors are discrete). However it fails miserably, when a smooth coloring is used, as while it is able to determine where the Mandelbrot Set is, for points outside the Mandelbrot Set since a smooth coloring is implemented and two pixels which have the same iteration count may not have the same color, the normal quadtree algorithm becomes forced to calculate each pixel and then determine whether to split, eventually splitting to quadtrees of single pixels. This process eventually becomes just less efficient than simple rasterization and just looping through the non calculated pixels. Thus in smooth coloring the mixed method reigns supreme, as it effectively blocks off the areas where the Mandelbrot set is and then rasterizes the remainder of the image thus eliminating unnecessary splits and more. While seeming less efficient than the other algorithms, rasterization performs the best when frame doesn’t include much of the Mandelbrot set, but if the Mandelbrot set is a good portion of the frame, this algorithm slows down quite rapidly having to do the entire calculation for the elements within the Mandelbrot set. Overall the performance of the algorithm is dependent on the choice of domain and the choice of coloring. However it is important to note that the most important areas to look at have non trivial amounts of the mandelbrot sets. So overall the mixed method performs the best.

While these optimizations do bring a boost in performance, they are still troubled by python's inherent slowness. Due to python being a interpreted language, there exists a inherent overhead associated with any program. So while pure python works as expected, it is very slow at high escape radius and max iteration values. This is why there is an Accelerated implementation of the set generator. The Accelerated implementation uses numba which just in time compiles several key functions to get a massive performance boost. To use numba you have to provide a static typing to your functions so it knows what to expect as a input and output. The performance boost numba provides is massive, however there are a few caveats (can't get a completely free lunch). Due to having to precompile the functions at runtime, it spends around 4 seconds compiling the functions and preparing them to run. So at small max iteration (< 50), the default escape radius, small image sizes, and default domain, numba may have equivalent or worse performance to the normal implementation. However at such small iteration values we lose out on precision and sharpness. However once we get out of this extremely idealized conditions, the benefits numba gives us are tremendous. In the spiral image below, the normal computation with a mixed quadtree algorithm took about 90+ seconds to compute while the numba accelerated implementation took only about 4.5 seconds to compute (4 seconds in compiling time and .5 seconds in actual generation time). However it is important to note that while the normal implementation is able to exploit symmetry with a dictionary, the accelerated implementation is not able to do so because the use of dictionaries in parallel tasks may cause memory issues.

The project uses a list (general data storage), tuples (coupling data together quickly), dictionary (preventing the unnecessary computation for Normal Mandelbrot Implementation), queue (Breath First Search Algorithm), and a tree (quadtree data structure). The project uses the following packages:
- numpy is used for fast math calculations
- yaml is used to parse yaml configuration file
- argparse is used to parse for command line arguments
- matplotlib is used for colormaps
- numba is used for fast math calculations, parallelization, and just in time compiling
- Pillow is used for image creation.

The main.py file is used to run the code with the user specified parameters.

## Future
Right now the generator only makes still images, but I want to eventually have it make zooming mandelbrot videos.

## Images
![Full Mandelbrot Set](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/grayscale/image.png?raw=true)

![enter image description here](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/quadtree_view/image.png?raw=true)


![enter image description here](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/alice_in_wonderland_normal/image.png?raw=true)

![enter image description here](https://github.com/arham-lodha/Mandelbrot-Generator/blob/master/examples/spiral_twilight/image.png?raw=true)


## References
Wikipedia about the mandelbrot set - https://en.wikipedia.org/wiki/Mandelbrot_set
Wikipedia about drawing algorithms for the Mandelbrot set - https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
Mu-Ency - The Encyclopedia of the Mandelbrot Set - https://www.mrob.com/pub/muency.html
Real Python Website about drawing Mandelbrot set - https://realpython.com/mandelbrot-set-python/#drawing-the-mandelbrot-set-with-pillow