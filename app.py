from shiny import App, render, ui, reactive
from pathlib import Path
import base64
import io
import numpy as np
from PIL import Image
from utils import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("i_images", "Sélectionnez vos images", accept=[".png", ".jpg", ".jpeg"], multiple=True),
        ui.hr(),
        ui.h3("Bruitage"),
        ui.input_numeric("i_bruit_sel_poivre", "Coefficient Sel & Poivre", value=0.25, min=0, max=1, step=0.01),
        ui.input_numeric("i_bruit_gaussien", "Coefficient Gaussien", value=0.1, min=0, max=1, step=0.01),
        ui.input_numeric("i_bruit_Speckel", "Coefficient Speckel", value=0.5, min=0, max=1, step=0.01),
        ui.hr(),
        ui.h3("Perona-Malik"),
        ui.input_numeric("i_niter", "Nombre d'itérations (Anisodiff)", value=75, min=1, step=1),
        ui.input_numeric("i_kappa", "Paramètre Kappa (Anisodiff)", value=50, min=1, step=1),
        ui.input_numeric("i_gamma", "Paramètre Gamma (Anisodiff)", value=0.1, min=0, step=0.01),
        ui.hr(),
        ui.h3("Coherence Enhancing"),
        ui.input_numeric("i_sigma", "Sigma", value=1, step=0.1, min=0.5, max=3),
        ui.input_numeric("i_str_sigma", "STR Sigma", value=11, min=0, max=1, step=0.01),
        ui.input_numeric("i_blend", "Bend", value=0.5, min=1, max=0.25, step=0.05),
        ui.input_numeric("i_iter_n", "Itérations", value=4, min=1, step=1),
        # ui.input_numeric("i_c", "C", value=1e-10, min=1, step=1),
        ui.hr(),
        ui.input_action_button("update_noise", "Mettre à jour le bruit"),
    ),
    ui.row(
        ui.column(3, "Image"),
        ui.column(3, "Bruit"),
        ui.column(3, "Malik & Perona"),
        ui.column(3, "Coherence Enhancing")
    ),
    ui.output_ui("resultats"),
    title="Lissage & Débruitage",
)

def server(input, output, session):
    noise_params = reactive.Value({"sp": 0.25, "gaussian": 0.25, "speckle": 0.25})
    anisodiff_params = reactive.Value({"niter": 1, "kappa": 50, "gamma": 0.1})
    coherence_enhancing_params = reactive.Value({"sigma": 1, "str_sigma": 11, "blend":0.5, "iter_n": 4})

    @reactive.Effect
    @reactive.event(input.update_noise)
    def _():
        noise_params.set({
            "sp": input.i_bruit_sel_poivre(),
            "gaussian": input.i_bruit_gaussien(),
            "speckle": input.i_bruit_Speckel()
        })
        anisodiff_params.set({
            "niter": input.i_niter(),
            "kappa": input.i_kappa(), 
            "gamma": input.i_gamma()
        })
        coherence_enhancing_params.set({
            "sigma": input.i_sigma(),
            "str_sigma": input.i_str_sigma(), 
            "blend": input.i_blend(),
            "iter_n": input.i_iter_n()
        })

    @output
    @render.ui
    @reactive.event(input.i_images, input.update_noise)
    def resultats():
        files = input.i_images()
        if files is None or len(files) < 1:
            return None
        
        images = []
        for file in files:
            file_path = Path(file["datapath"])
            with open(file_path, "rb") as img_file:
                img = Image.open(img_file).convert('L')  # Convert to grayscale
                img_array = np.array(img)
                
                # Apply noise to the image
                noisy_img_sp = add_salt_and_pepper_noise(img, noise_params.get()["sp"])
                noisy_img_gaussian = add_gaussian_noise(img, noise_params.get()["gaussian"])
                noisy_img_speckle = add_speckle_noise(img, noise_params.get()["speckle"])
                
                # Apply anisotropic diffusion to the noisy images
                img_array_sp = np.array(noisy_img_sp)
                img_array_gaussian = np.array(noisy_img_gaussian)
                img_array_speckle = np.array(noisy_img_speckle)

                anisodiff_img_sp = anisodiff(img_array_sp, 
                                             niter=anisodiff_params.get()["niter"],
                                             kappa=anisodiff_params.get()["kappa"],
                                             gamma=anisodiff_params.get()["gamma"])
                anisodiff_img_gaussian = anisodiff(img_array_gaussian, 
                                                   niter=anisodiff_params.get()["niter"],
                                                   kappa=anisodiff_params.get()["kappa"],
                                                   gamma=anisodiff_params.get()["gamma"])
                anisodiff_img_speckle = anisodiff(img_array_speckle, 
                                                  niter=anisodiff_params.get()["niter"],
                                                  kappa=anisodiff_params.get()["kappa"],
                                                  gamma=anisodiff_params.get()["gamma"])
                

                # filtered_img = coherence_filter_image(img, sigma=11, str_sigma=11, blend=0.5, iter_n=4)
                coherence_enhancing_img_sp = coherence_filter_image(img_array_sp,
                    sigma= coherence_enhancing_params.get()["sigma"],
                    str_sigma= coherence_enhancing_params.get()["str_sigma"], 
                    blend= coherence_enhancing_params.get()["blend"],
                    iter_n= coherence_enhancing_params.get()["iter_n"]
                )
                coherence_enhancing_img_gaussian = coherence_filter_image(img_array_gaussian,
                    sigma= coherence_enhancing_params.get()["sigma"],
                    str_sigma= coherence_enhancing_params.get()["str_sigma"], 
                    blend= coherence_enhancing_params.get()["blend"],
                    iter_n= coherence_enhancing_params.get()["iter_n"]
                )
                coherence_enhancing_img_speckle = coherence_filter_image(img_array_speckle,
                    sigma= coherence_enhancing_params.get()["sigma"],
                    str_sigma= coherence_enhancing_params.get()["str_sigma"], 
                    blend= coherence_enhancing_params.get()["blend"],
                    iter_n= coherence_enhancing_params.get()["iter_n"]
                )

                # Calculate scores
                def calculate_scores(original, compared):
                    psnr_score = psnr(original, compared)
                    mse_score = mse(original, compared)
                    ssim_score, _ = ssim(original, compared, full=True)
                    return f"<br>PSNR: {psnr_score:.2f}<br>MSE: {mse_score:.2f}<br>SSIM: {ssim_score:.2f}"

                sp_scores = calculate_scores(img_array, img_array_sp)
                gaussian_scores = calculate_scores(img_array, img_array_gaussian)
                speckle_scores = calculate_scores(img_array, img_array_speckle)

                sp_malik_scores = calculate_scores(img_array, anisodiff_img_sp)
                gaussian_malik_scores = calculate_scores(img_array, anisodiff_img_gaussian)
                speckle_malik_scores = calculate_scores(img_array, anisodiff_img_speckle)

                sp_coherence_enhancing_scores = calculate_scores(img_array, coherence_enhancing_img_sp)
                gaussian_coherence_enhancing_scores = calculate_scores(img_array, coherence_enhancing_img_gaussian)
                speckle_coherence_enhancing_scores = calculate_scores(img_array, coherence_enhancing_img_speckle)

                # Convert images to base64
                def image_to_base64(image):
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    return base64.b64encode(buffered.getvalue()).decode()

                img_base64 = image_to_base64(img)
                noisy_img_sp_base64 = image_to_base64(noisy_img_sp)
                noisy_img_gaussian_base64 = image_to_base64(noisy_img_gaussian)
                noisy_img_speckle_base64 = image_to_base64(noisy_img_speckle)

                anisodiff_img_sp_pil = Image.fromarray(anisodiff_img_sp)
                anisodiff_img_gaussian_pil = Image.fromarray(anisodiff_img_gaussian)
                anisodiff_img_speckle_pil = Image.fromarray(anisodiff_img_speckle)

                anisodiff_img_sp_base64 = image_to_base64(anisodiff_img_sp_pil)
                anisodiff_img_gaussian_base64 = image_to_base64(anisodiff_img_gaussian_pil)
                anisodiff_img_speckle_base64 = image_to_base64(anisodiff_img_speckle_pil)

                coherence_enhancing_img_sp_pil = Image.fromarray(coherence_enhancing_img_sp)
                coherence_enhancing_img_gaussian_pil = Image.fromarray(coherence_enhancing_img_gaussian)
                coherence_enhancing_img_speckle_pil = Image.fromarray(coherence_enhancing_img_speckle)

                coherence_enhancing_img_sp_base64 = image_to_base64(coherence_enhancing_img_sp_pil)
                coherence_enhancing_img_gaussian_base64 = image_to_base64(coherence_enhancing_img_gaussian_pil)
                coherence_enhancing_img_speckle_base64 = image_to_base64(coherence_enhancing_img_speckle_pil)

            images.append(
                ui.div(
                    ui.row(
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{img_base64}", style="max-width: 200px; margin: 10px;")),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{noisy_img_sp_base64}", style="max-width: 200px; margin: 10px;"),ui.HTML(sp_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_sp_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(sp_malik_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{coherence_enhancing_img_sp_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(sp_coherence_enhancing_scores)),    
                    ),
                    ui.row(
                        ui.column(3, ""),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{noisy_img_gaussian_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(gaussian_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_gaussian_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(gaussian_malik_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{coherence_enhancing_img_gaussian_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(gaussian_coherence_enhancing_scores))     
                    ),
                    ui.row(
                        ui.column(3, ""),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{noisy_img_speckle_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(speckle_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_speckle_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(speckle_malik_scores)),
                        ui.column(3, ui.tags.img(src=f"data:image/png;base64,{coherence_enhancing_img_speckle_base64}", style="max-width: 200px; margin: 10px;"), ui.HTML(speckle_coherence_enhancing_scores))     
                    )
                )  
            )            
        
        return ui.div(images)

app = App(app_ui, server)
