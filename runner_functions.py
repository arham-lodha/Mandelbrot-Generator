import time


def create_image(config, directory):
    if config['accelerated']:
        tic = time.perf_counter()

        from Accelerated.accelerated_mandelbrot import AcceleratedMandelbrot

        toc = time.perf_counter()

        print(f"Finished compiling in {toc - tic:0.4f} seconds")

        mandelbrot = AcceleratedMandelbrot(
            size=(config["width"], config["height"]),
            center=config['center'],
            width=config['world_width'],
            max_iterations=config['max_iterations'],
            escape_radius=config['escape_radius'],
            color_scheme=config['color_scheme'],
            color_map=config['colormap'],
            smooth=config['smooth'],
            raster=config['raster'],
            mixed_raster=config['mixed_raster'],
            fast_quadtree=config['fast_quadtree'],
            show_quadtree=config['show_quadtree']
        )

        tic = time.perf_counter()

        mandelbrot.generate()

        toc = time.perf_counter()

        print(f"Finished calculating in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()

        mandelbrot.render(f"{directory}/image.png")

        toc = time.perf_counter()

        print(f"Finished rendering and saving in {toc - tic:0.4f} seconds")
    else:
        from Normal.mandelbrot import Mandelbrot
        from Normal.coloring import color_scheme, generate_colormap_coloring

        if config['color_scheme'] == 4:
            scheme = generate_colormap_coloring(config['color_scheme'])
        else:
            scheme = color_scheme[config['color_scheme']]

        mandelbrot = Mandelbrot(
            size=(config["width"], config["height"]),
            center=config['center'],
            width=config['world_width'],
            max_iterations=config['max_iterations'],
            escape_radius=config['escape_radius'],
            color=scheme,
            smooth=config['smooth'],
            raster=config['raster'],
            mixed_raster=config['mixed_raster'],
            show_quadtree=config['show_quadtree']
        )

        tic = time.perf_counter()

        mandelbrot.generate()

        toc = time.perf_counter()

        print(f"Finished calculating in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()

        mandelbrot.render(f"{directory}/image.png")

        toc = time.perf_counter()

        print(f"Finished rendering and saving in {toc - tic:0.4f} seconds")
