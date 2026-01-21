#!/usr/bin/env python3
"""
Script per visualizzare le mode rilevate sul skeleton in modo pulito.
Versione semplificata senza punti, legenda e statistiche - solo il skeleton colorato.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def load_skeleton_data(npz_path):
    """Carica i dati del skeleton da file .npz"""
    data = np.load(npz_path)
    return {
        'skeleton': data['skeleton'],
        'distances': data['distances'],
        'local_widths': data['local_widths'],
        'separating_spaces': data['separating_spaces']
    }

def detect_modes_from_widths(local_widths, skeleton):
    """
    Rileva le mode dalle larghezze locali del skeleton.
    Simula l'algoritmo usato nello script principale.
    """
    # Estrai le larghezze dei punti dello skeleton
    skeleton_coords = np.where(skeleton)
    if len(skeleton_coords[0]) == 0:
        return [], []
    
    skeleton_widths = local_widths[skeleton_coords[0], skeleton_coords[1]]
    
    # Filtra larghezze valide (maggiore di 0)
    valid_widths = skeleton_widths[skeleton_widths > 0]
    
    if len(valid_widths) < 10:
        return [], []
    
    # Prova con 3 mode
    from sklearn.mixture import GaussianMixture
    
    try:
        gmm_3 = GaussianMixture(n_components=3, random_state=42)
        gmm_3.fit(valid_widths.reshape(-1, 1))
        means_3 = gmm_3.means_.flatten()
        stds_3 = np.sqrt(gmm_3.covariances_.flatten())
        
        # Controlla separazione delle mode
        sorted_indices = np.argsort(means_3)
        means_sorted = means_3[sorted_indices]
        stds_sorted = stds_3[sorted_indices]
        
        # Calcola separazione tra mode consecutive
        separations = []
        for i in range(len(means_sorted) - 1):
            separation = (means_sorted[i+1] - means_sorted[i]) / (stds_sorted[i] + stds_sorted[i+1])
            separations.append(separation)
        
        min_separation_3 = min(separations) if separations else 0
        
        # Se la separazione minima è buona, usa 3 mode
        if min_separation_3 >= 1.5:
            modes = means_sorted
            mode_stds = stds_sorted
            n_modes = 3
        else:
            # Prova con 2 mode
            gmm_2 = GaussianMixture(n_components=2, random_state=42)
            gmm_2.fit(valid_widths.reshape(-1, 1))
            means_2 = gmm_2.means_.flatten()
            stds_2 = np.sqrt(gmm_2.covariances_.flatten())
            
            sorted_indices_2 = np.argsort(means_2)
            modes = means_2[sorted_indices_2]
            mode_stds = stds_2[sorted_indices_2]
            n_modes = 2
            
    except:
        # Fallback: usa 2 mode
        try:
            gmm_2 = GaussianMixture(n_components=2, random_state=42)
            gmm_2.fit(valid_widths.reshape(-1, 1))
            means_2 = gmm_2.means_.flatten()
            stds_2 = np.sqrt(gmm_2.covariances_.flatten())
            
            sorted_indices_2 = np.argsort(means_2)
            modes = means_2[sorted_indices_2]
            mode_stds = stds_2[sorted_indices_2]
            n_modes = 2
        except:
            modes = []
            mode_stds = []
            n_modes = 0
    
    return modes, mode_stds

def create_clean_mode_visualization(skeleton_data, filename, name):
    """Crea una visualizzazione pulita del skeleton con colori basati sulle mode"""
    
    # Crea la figura con sfondo bianco
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')
    
    # Mostra il skeleton originale
    skeleton = skeleton_data['skeleton']
    distances = skeleton_data['distances']
    
    # Rileva le mode
    modes, mode_stds = detect_modes_from_widths(skeleton_data['local_widths'], skeleton)
    
    if len(modes) > 0:
        print(f"Detected {len(modes)} modes: {modes}")
        
        # Crea un'immagine RGB per il skeleton con colori basati sulle mode
        skeleton_rgb = np.zeros((skeleton.shape[0], skeleton.shape[1], 3))
        
        # Colora i punti del skeleton in base alla vicinanza alle mode
        skeleton_coords = np.where(skeleton)
        for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
            local_width = distances[y, x] * 2
            
            # Trova la mode più vicina
            distances_to_modes = [abs(local_width - mode) for mode in modes]
            closest_mode_idx = np.argmin(distances_to_modes)
            
            # Assegna colori basati sulla mode più vicina
            if closest_mode_idx == 0:
                skeleton_rgb[y, x] = [1, 0, 0]  # Rosso per prima mode
            elif closest_mode_idx == 1:
                skeleton_rgb[y, x] = [0, 0, 1]  # Blu per seconda mode
            else:
                skeleton_rgb[y, x] = [0, 1, 0]  # Verde per terza mode
    
    else:
        # Fallback: usa colori basati sulla larghezza
        skeleton_rgb = np.zeros((skeleton.shape[0], skeleton.shape[1], 3))
        skeleton_coords = np.where(skeleton)
        for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
            local_width = distances[y, x] * 2
            
            if local_width < 5:
                skeleton_rgb[y, x] = [1, 0, 0]  # Rosso per stretto
            elif local_width < 15:
                skeleton_rgb[y, x] = [0, 0, 1]  # Blu per medio
            else:
                skeleton_rgb[y, x] = [0, 1, 0]  # Verde per largo
    
    # Mostra il skeleton senza bordi
    ax.imshow(skeleton_rgb, origin='lower')
    
    # Rimuovi tutti gli elementi decorativi
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Rimuovi titoli e etichette
    ax.set_title('', fontsize=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Imposta i margini a zero
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Salva l'immagine senza bordi
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                pad_inches=0)
    plt.close()

def process_all_files():
    """Processa tutti i file .npz nella cartella skeleton_analysis"""
    input_dir = "../2. skeleton_analysis"
    output_dir = "mode_clean"
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file .npz
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('_skeleton_data.npz')]
    npz_files.sort()
    
    print(f"Creating clean mode visualizations from {len(npz_files)} files...")
    
    for npz_file in npz_files:
        name = npz_file.replace('_skeleton_data.npz', '')
        npz_path = os.path.join(input_dir, npz_file)
        
        print(f"Processing {name}...")
        
        try:
            # Carica i dati
            skeleton_data = load_skeleton_data(npz_path)
            
            # Crea visualizzazione pulita
            png_filename = os.path.join(output_dir, f"{name}_clean_mode.png")
            create_clean_mode_visualization(skeleton_data, png_filename, name)
            
            print(f"  Created: {png_filename}")
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue
    
    print("Clean mode visualization complete!")

def process_single_file(file_number):
    """Processa un singolo file specifico"""
    input_dir = "../2. skeleton_analysis"
    output_dir = "mode_clean"
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    npz_file = f"{file_number}_skeleton_data.npz"
    npz_path = os.path.join(input_dir, npz_file)
    
    if not os.path.exists(npz_path):
        print(f"File {npz_path} not found!")
        return
    
    print(f"Processing {file_number}...")
    
    try:
        # Carica i dati
        skeleton_data = load_skeleton_data(npz_path)
        
        # Crea visualizzazione pulita
        png_filename = os.path.join(output_dir, f"{file_number}_clean_mode.png")
        create_clean_mode_visualization(skeleton_data, png_filename, str(file_number))
        
        print(f"  Created: {png_filename}")
        
    except Exception as e:
        print(f"  Error processing {file_number}: {e}")

if __name__ == "__main__":
    import sys
    
    # Cambia directory di lavoro alla cartella dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if len(sys.argv) > 1:
        # Processa un singolo file se specificato
        file_number = sys.argv[1]
        process_single_file(file_number)
    else:
        # Processa tutti i file
        process_all_files()
