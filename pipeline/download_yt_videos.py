import os
import yt_dlp

### Función principal para descargar videos de youtube en formato mp4

def download_yt_videos_list(list_videos_urls, folder_destination):
    if folder_destination and not os.path.exists(folder_destination):
        os.makedirs(folder_destination)

    for video_link in list_videos_urls:
        if folder_destination:
            outtmpl = os.path.join(folder_destination, '%(title)s.%(ext)s')
        else:
            outtmpl = '%(title)s.%(ext)s'
        ## En este diccionario se han puesto la configuración para la librería    
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'noplaylist': True,
            'outtmpl': outtmpl,
            'quiet': True,
            'no_warnings': True,
        }
        ## Salvar los videos en el folder seleccionado
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_link, download=True)
                video_title = info_dict.get('title', 'video')
        except yt_dlp.utils.DownloadError as e:
            print(f"Un error ha ocurrido al descargar el link {video_link}: {e}")
        except Exception as e:
            print(f"Error inesperado al descargar el link del video {video_link}: {e}")

