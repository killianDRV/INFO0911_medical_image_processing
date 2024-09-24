from shiny import App, render, ui, reactive
from pathlib import Path
import base64
import io
import numpy as np
from PIL import Image
from utils import *

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("i_images", "Sélectionnez vos images", accept=[".png", ".jpg", ".jpeg"], multiple=True),
        ui.input_numeric("i_bruit_sel_poivre", "Coefficient Sel & Poivre", value=0.25, min=0, max=1, step=0.01),
        ui.input_numeric("i_bruit_gaussien", "Coefficient Gaussien", value=0.1, min=0, max=1, step=0.01),
        ui.input_numeric("i_bruit_Speckel", "Coefficient Speckel", value=0.5, min=0, max=1, step=0.01),
        ui.input_numeric("i_niter", "Nombre d'itérations (Anisodiff)", value=75, min=1, step=1),
        ui.input_numeric("i_kappa", "Paramètre Kappa (Anisodiff)", value=50, min=1, step=1),
        ui.input_numeric("i_gamma", "Paramètre Gamma (Anisodiff)", value=0.1, min=0, step=0.01),
        ui.input_action_button("update_noise", "Mettre à jour le bruit"),
    ),
    ui.row(
        ui.column(4, "Image"),
        ui.column(4, "Bruit"),
        ui.column(4, "Malik & Perona")
    ),
    ui.output_ui("resultats"),
    title="Lissage & Débruitage",
)

def server(input, output, session):
    noise_params = reactive.Value({"sp": 0.25, "gaussian": 0.25, "speckle": 0.25})
    anisodiff_params = reactive.Value({"niter": 1, "kappa": 50, "gamma": 0.1})

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

            images.append(
                ui.div(
                    ui.row(
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{img_base64}", style="max-width: 200px; margin: 10px;")),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{noisy_img_sp_base64}", style="max-width: 200px; margin: 10px;")),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_sp_base64}", style="max-width: 200px; margin: 10px;"))    
                    ),
                    ui.row(
                        ui.column(4, ""),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{noisy_img_gaussian_base64}", style="max-width: 200px; margin: 10px;")),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_gaussian_base64}", style="max-width: 200px; margin: 10px;"))    
                    ),
                    ui.row(
                        ui.column(4, ""),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{noisy_img_speckle_base64}", style="max-width: 200px; margin: 10px;")),
                        ui.column(4, ui.tags.img(src=f"data:image/png;base64,{anisodiff_img_speckle_base64}", style="max-width: 200px; margin: 10px;"))    
                    )
                )  
            )            
        
        return ui.div(images)

app = App(app_ui, server)
