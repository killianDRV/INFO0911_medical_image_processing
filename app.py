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

def calculate_scores(original, compared):
    psnr_score = psnr(original, compared)
    mse_score = mse(original, compared)
    ssim_score, _ = ssim(original, compared, full=True)
    return f"<br>PSNR: {psnr_score:.2f}<br>MSE: {mse_score:.2f}<br>SSIM: {ssim_score:.2f}"

def cv2_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def cv2_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')
    
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("i_images", "Sélectionnez vos images", accept=[".png", ".jpg", ".jpeg"], multiple=True),
        ui.hr(),
        ui.input_selectize(
            "method_choice",
            "Choisissez une ou plusieurs méthodes :",
            {
                "bruitage": "Bruitage",
                "perona_malik": "Perona-Malik",
                "coherence_enhancing": "Coherence Enhancing",
                "find_contours" : "Find Contours",
            },
            multiple=True,
        ),
        ui.output_ui("dynamic_sidebar"),
        ui.input_action_button("update_noise", "Mettre à jour le bruit"),
    ),
    ui.output_ui("resultats"),
    title="Lissage & Débruitage",
)

def server(input, output, session):
    @output
    @render.ui
    def dynamic_bruitage():
        selected_bruitage = input.i_bruitage()   
        if selected_bruitage == "sel_poivre":
            return ui.input_numeric("i_bruit_coef", "Coefficient Sel & Poivre", value=0.25, min=0, max=1, step=0.01)
        elif selected_bruitage == "gaussien":
            return ui.input_numeric("i_bruit_coef", "Coefficient Gaussien", value=0.1, min=0, max=1, step=0.01)
        elif selected_bruitage == "coefficient_speckel":
            return  ui.input_numeric("i_bruit_coef", "Coefficient Speckel", value=0.5, min=0, max=1, step=0.01)
      
    @output
    @render.ui
    def dynamic_sidebar():
        # Liste des éléments dynamiques
        elements = []
        selected_methods = input.method_choice()

        # Ajouter des champs dynamiquement en fonction de la sélection
        for method in selected_methods:
            if method == "bruitage":
                elements.extend([
                    ui.h3("Bruitage"),
                    ui.input_select("i_bruitage",
                    "Méthode de Bruitage", 
                    {
                        "sel_poivre": "Sel & Poivre",
                        "gaussien": "Gaussien",
                        "coefficient_speckel": "Coefficient Speckel",
                    } 
                ),
                ui.output_ui("dynamic_bruitage"),
                ui.hr(),
                ])
            
            elif method == "perona_malik":
                elements.extend([
                    ui.h3("Perona-Malik"),
                    ui.input_numeric("i_niter", "Nombre d'itérations (Anisodiff)", value=75, min=1, step=1),
                    ui.input_numeric("i_kappa", "Paramètre Kappa (Anisodiff)", value=50, min=1, step=1),
                    ui.input_numeric("i_gamma", "Paramètre Gamma (Anisodiff)", value=0.1, min=0, step=0.01),
                    ui.hr(),
                ])
            
            elif method == "coherence_enhancing":
                elements.extend([
                    ui.h3("Coherence Enhancing"),
                    ui.input_numeric("i_sigma", "Sigma", value=1, step=2, min=1, max=31),
                    ui.input_numeric("i_str_sigma", "STR Sigma", value=2, min=1, max=10, step=1),
                    ui.input_numeric("i_blend", "Bend", value=0.5, min=0.25, max=1, step=0.05),
                    ui.input_numeric("i_iter_n", "Itérations", value=4, min=1, step=1),
                    # ui.input_numeric("i_c", "C", value=1e-10, min=1, step=1),
                    ui.hr(),
                ])
            
            elif method == "find_contours":
                elements.extend([
                    ui.h3("Find Contours"),
                    ui.input_numeric("i_contours_min", "Seuil min", value=35, step=1, min=0, max=100),
                    ui.input_numeric("i_contours_max", "Seuil max", value=40, step=1, min=0, max=100)
                ])
        
        # Retourner les éléments générés dynamiquement
        return elements
        
    @output
    @render.ui
    @reactive.event(input.i_images, input.update_noise)
    def resultats():
        files = input.i_images()
        if files is None or len(files) < 1:
            return None
        
        selected_methods = input.method_choice()
        if "find_contours" in selected_methods:
            threshold_min = input.i_contours_min()
            threshold_max = input.i_contours_max()
        if "coherence_enhancing" in selected_methods :
            sigma = input.i_sigma()
            str_sigma = input.i_str_sigma()
            blend = input.i_blend()
            iter_n = input.i_iter_n()
        
        if "perona_malik" in selected_methods:
            niter = input.i_niter()
            kappa = input.i_kappa() 
            gamma = input.i_gamma()

        if "bruitage" in selected_methods:
            bruit = input.i_bruitage()
            bruit_coef = input.i_bruit_coef()
       
        images = []
        columns = []
        col_length = int(min(12/(max(len(selected_methods),1)+1),12))

        # Ajout des en-têtes des méthodes sélectionnées        
        columns.append(ui.column(col_length, "Image"))
        for method in selected_methods:
            columns.append(ui.column(col_length, method))
        images.append(ui.div(ui.row((columns))))

        for file in files:
            cv2_img = formate_image(cv2.imread(Path(file["datapath"]), cv2.IMREAD_GRAYSCALE)) 
            columns = []
            img_before = cv2_img
            columns.append(ui.column(col_length, ui.tags.img(src=f"data:image/png;base64,{cv2_to_base64(cv2_img)}", style="max-width: 200px; margin: 10px;")))
            for i in range(0, len(selected_methods)):
                method = selected_methods[i]

                # BRUITAGE
                if method == "bruitage":
                    if input.i_bruitage() == "sel_poivre":
                        img_after = np.array(add_salt_and_pepper_noise(img_before, bruit_coef))
                    elif input.i_bruitage() == "gaussien":
                        img_after = np.array(add_gaussian_noise(img_before, bruit_coef))
                    elif input.i_bruitage() == "coefficient_speckel":
                        img_after = np.array(add_speckle_noise(img_before, bruit_coef)) 

                # PERONA-MALIK
                if method == "perona_malik":
                    img_after = anisodiff(img_before, niter, kappa, gamma)            

                # COHERENCE-ENHANCING
                if method == "coherence_enhancing":
                    img_after = coherence_filter_image(img_before,
                        sigma,
                        str_sigma, 
                        blend,
                        iter_n
                    )

                # FIND-CONTOURS
                # if method == "find_contours":
                #     rows = []
                #     column = []
                #     for threshold in range(find_contours_params.get()["threshold_min"], find_contours_params.get()["threshold_max"]):
                #         img_after = find_contours(img_before, threshold)
                #         sp_scores = calculate_scores(cv2_img, img_after)
                #         rows.append(ui.row(ui.tags.img(src=f"data:image/png;base64,{cv2_to_base64(img_after)}", style="max-width: 200px; margin: 10px;"),ui.HTML(sp_scores + "\n test")))
                #     column.append(rows)
                #     columns.append(column)

                # FIND-CONTOURS
                if method == "find_contours":
                    # Crée une liste pour stocker les lignes correspondant aux différentes images générées
                    contour_rows = []
                    
                    for threshold in range(threshold_min, threshold_max+1):
                        img_after = find_contours(img_before, threshold)

                        # Ajout d'une ligne avec l'image et le seuil correspondant
                        contour_rows.append(
                            ui.row(
                                ui.column(
                                    12, 
                                    ui.tags.img(src=f"data:image/png;base64,{cv2_to_base64(img_after)}", style="max-width: 200px; margin: 10px;"),
                                    ui.HTML(f"<div>Threshold: {threshold}</div>")
                                )
                            )
                        )
                    
                    # Ajoute une seule colonne contenant toutes les lignes générées
                    columns.append(ui.column(col_length, ui.div(contour_rows)))


                if method != "find_contours":
                    sp_scores = calculate_scores(cv2_img, img_after)
                    columns.append(ui.column(col_length, ui.tags.img(src=f"data:image/png;base64,{cv2_to_base64(img_after)}", style="max-width: 200px; margin: 10px;"),ui.HTML(sp_scores)))
                    img_before = img_after

            images.append(
                ui.div(
                    ui.row(
                        (columns)
                    )
                )
            )
        return ui.div(images)

app = App(app_ui, server)
