from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.camera import Camera
from kivy.core.window import Window
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image as PILImage

# Modeli yükle
model = load_model('bitki_hastaligi_model.keras')

# Arka plan rengini yeşil yapıyoruz
Window.clearcolor = (0, 1, 0, 1)  # RGBA formatında

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        
        layout = BoxLayout(orientation='vertical')
        
        # GreenAI logosu ekleniyor
        logo = Image(source='greenai_logo.png', size_hint=(1, 0.3))
        layout.add_widget(logo)
        
        # Butonları ortalamak için dış layout
        outer_layout = BoxLayout(size_hint=(1, 0.7))
        
        # Butonları ortalamak için iç layout
        button_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        
        self.camera_button = Button(text="Fotoğraf Çek", size_hint=(1, 0.2))
        self.camera_button.bind(on_press=self.capture_photo)
        button_layout.add_widget(self.camera_button)
        
        self.gallery_button = Button(text="Galeriden Yükle", size_hint=(1, 0.2))
        self.gallery_button.bind(on_press=self.load_from_gallery)
        button_layout.add_widget(self.gallery_button)
        
        outer_layout.add_widget(button_layout)
        layout.add_widget(outer_layout)
        
        self.add_widget(layout)
    
    def capture_photo(self, instance):
        App.get_running_app().root.current = 'camera'
    
    def load_from_gallery(self, instance):
        App.get_running_app().root.current = 'filechooser'

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        self.camera = Camera(play=True)
        layout.add_widget(self.camera)
        
        self.capture_button = Button(text="Fotoğraf Çek", size_hint=(1, 0.1))
        self.capture_button.bind(on_press=self.capture)
        layout.add_widget(self.capture_button)
        
        self.add_widget(layout)
        
    def capture(self, instance):
        self.camera.export_to_png("captured_image.png")
        App.get_running_app().photo_path = "captured_image.png"
        App.get_running_app().predict_and_set_label()
        App.get_running_app().root.current = 'photo'

class FileChooserScreen(Screen):
    def __init__(self, **kwargs):
        super(FileChooserScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        self.filechooser = FileChooserIconView()
        self.filechooser.bind(on_selection=self.selected)
        layout.add_widget(self.filechooser)
        
        self.add_widget(layout)
    
    def selected(self, filechooser, selection):
        if selection:
            App.get_running_app().photo_path = selection[0]
            App.get_running_app().predict_and_set_label()
            App.get_running_app().root.current = 'photo'

class PhotoScreen(Screen):
    def __init__(self, **kwargs):
        super(PhotoScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        self.image = Image()
        layout.add_widget(self.image)
        
        self.label = Label(text="Hello, World!")
        layout.add_widget(self.label)
        
        self.add_widget(layout)
    
    def on_pre_enter(self, *args):
        self.image.source = App.get_running_app().photo_path
        self.label.text = App.get_running_app().prediction_text

class PhotoApp(App):
    def build(self):
        self.photo_path = ""
        self.prediction_text = "Hello, World!"
        
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(FileChooserScreen(name='filechooser'))
        sm.add_widget(PhotoScreen(name='photo'))
        
        return sm
    
    def predict_and_set_label(self):
        image = PILImage.open(self.photo_path).convert("RGB")  # RGBA'dan RGB'ye dönüştür
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype("float32")
        
        tahmin = model.predict(image)
        en_yuksek_olasilik_sinifi = np.argmax(tahmin)
        sinif_etiketleri = ["Bacteria", "Fungi", "Healthy", "Pests", "Virus"]
        tahmin_etiket = sinif_etiketleri[en_yuksek_olasilik_sinifi]
        
        # Tahmin sonucunu sakla
        self.prediction_text = f"Tahmin edilen hastalık: {tahmin_etiket}"
        
        # AI açıklaması ekleme
        try:
            from g4f.client import Client
            
            chat = f"Bitkimde ki {tahmin_etiket} sorununu çözmek için izleyeceğim adımlar nelerdir."
            client = Client()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": chat}],
            )
            ai_response = response.choices[0].message.content
            self.prediction_text += f"\n\nAI Cevabı: {ai_response}"
        except Exception as e:
            self.prediction_text += f"\n\nAI Cevabı alınırken hata oluştu: {e}"

if __name__ == '__main__':
    PhotoApp().run()
