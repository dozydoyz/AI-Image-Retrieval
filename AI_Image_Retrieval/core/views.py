from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from .forms import UploadForm
from .models import SearchHistory
from .services import SearchEngine
from .classifier import ImageClassifier

@login_required
def index(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            record = form.save(commit=False)
            record.user = request.user 
            record.save()
            # 搜图(基于DINOv2)
            engine = SearchEngine.get_instance()
            results = engine.search(record.image.path)
            cleaned_results = []
            for path, score in results:
                new_path = str(path).replace('\\', '/')
                if new_path.startswith('static/'):
                    new_path = new_path.replace('static/', '', 1)
                elif new_path.startswith('/static/'):
                    new_path = new_path.replace('/static/', '', 1)
                if new_path.startswith('/'):
                    new_path = new_path[1:]
                    
                cleaned_results.append((new_path, score))
            # 添加Lebal(基于ResNet50)
            classifier = ImageClassifier.get_instance()
            # 获取Top-5预测结果
            ai_results = classifier.predict(record.image.path, topk=5)
            return render(request, 'core/results.html', {
                'query_img': record.image,
                'results': cleaned_results,
                'ai_results': ai_results 
            })
    else:
        form = UploadForm()
    
    return render(request, 'core/index.html', {'form': form})

@login_required
def search_again(request, pk):
    """
    历史记录可以直接点击进行重新搜索
    """
    record = get_object_or_404(SearchHistory, pk=pk, user=request.user)
    
    # 搜索过程和直接搜索一样
    engine = SearchEngine.get_instance()
    results = engine.search(record.image.path)
    cleaned_results = []
    for path, score in results:
        new_path = path.replace('\\', '/')
        if new_path.startswith('static/'):
            new_path = new_path.replace('static/', '', 1)
        elif new_path.startswith('/static/'):
            new_path = new_path.replace('/static/', '', 1)
        if new_path.startswith('/'):
            new_path = new_path[1:]
        cleaned_results.append((new_path, score))
   
    classifier = ImageClassifier.get_instance()
    ai_results = classifier.predict(record.image.path, topk=5)
    
    return render(request, 'core/results.html', {
        'query_img': record.image,
        'results': cleaned_results,
        'ai_results': ai_results
    })

@login_required
def history(request):
    """
    历史记录页面
    """
    # 搜索历史按时间倒序排列
    records = SearchHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'core/history.html', {'records': records})

def register(request):
    """
    用户注册页面
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # 注册成功后直接登录
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})