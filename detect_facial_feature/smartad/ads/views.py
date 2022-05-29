from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import  render
from django.core.files.storage import FileSystemStorage
from ml import facial_feature

def index(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.delete('image.jpg')
        file = fss.save('image.jpg', upload)
        file_url = fss.url(file)
        return render(request, 'index.html', {'file_url': file_url})
    return render(request, 'index.html')
    
def predict_ads(request):
    res = {}
    try:
        print('operation starts')
        res = facial_feature.predict('media/image.jpg')
    except ValueError:
        print("some error occured")
    print(res)
    return render(request, 'result.html',res)