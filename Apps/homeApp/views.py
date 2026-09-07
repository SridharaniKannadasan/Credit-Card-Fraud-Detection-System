import io
import pickle
import pandas as pd
from django.shortcuts import render,redirect,get_object_or_404
from django.http.response import HttpResponseRedirect,HttpResponse,JsonResponse
from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import DataFileUpload

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score


def load_csv_data(source):
    try:
        if hasattr(source, 'seek'):
            source.seek(0)
        return pd.read_csv(source, sep=None, engine='python')
    except Exception:
        if hasattr(source, 'seek'):
            source.seek(0)
        return pd.read_csv(source, sep=';')

def base(request):
    return render(request,'homeApp/landing_page.html')
    
def upload_credit_data(request):
    return render(request,'homeApp/upload_credit_data.html')

def prediction_button(request,id):
    return render(request,'homeApp/fraud_detection.html', {'id': id})
    
def reports(request):
    all_data_files_objs=DataFileUpload.objects.all()
    return render(request,'homeApp/reports.html',{'all_files':all_data_files_objs})
    
def enter_form_data_manually(request):
    return render(request,'homeApp/enter_form_data_manually.html')

def predict_data_manually(request):
    return render(request,'homeApp/predict_data_manually.html')

def add_files_single(request, id):
    return render(request,'homeApp/add_files_single.html', {'id': id})

def predict_csv_single(request, id):
    if request.method == 'POST':
        try:
            obj = DataFileUpload.objects.get(id=id)
            loaded_model = pickle.loads(obj.trained_model_data)
            X_test = pickle.loads(obj.x_test_data)
            y_test = pickle.loads(obj.y_test_data)

            # precision score from training data
            y_pred = loaded_model.predict(X_test)
            precision = precision_score(y_test, y_pred, zero_division=0)

            uploaded_file = request.FILES['actual_file_name']
            # ✅ Convert uploaded file bytes to string for pandas
            df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')), sep=None, engine='python')

            # Drop Class column if exists
            if 'Class' in df.columns:
                X_input = df.drop('Class', axis=1)
            else:
                X_input = df

            # Scale using fresh scaler (same structure)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_input)

            prediction = loaded_model.predict(X_scaled)
            status = "Fraudulent Transaction" if prediction[0] == 1 else "Non-Fraudulent Transaction"

            context = {
                'status': status,
                'precision': f'{precision:.2%}',
                'data': df.iloc[0].to_dict()
            }
            return render(request, 'homeApp/predict_csv_single.html', context)

        except Exception as e:
            messages.warning(request, f"Invalid/wrong format: {e}")
            return redirect(f'/add_files_single/{id}')



def add_files_multi(request, id):
    return render(request,'homeApp/add_files_multi.html', {'id': id})
    
def predict_csv_multi(request, id):
    if request.method == 'POST':
        try:
            obj = DataFileUpload.objects.get(id=id)
            loaded_model = pickle.loads(obj.trained_model_data)
            X_test = pickle.loads(obj.x_test_data)
            y_test = pickle.loads(obj.y_test_data)

            y_pred = loaded_model.predict(X_test)
            precision = precision_score(y_test, y_pred, zero_division=0)

            uploaded_file = request.FILES['actual_file_name']
            # ✅ Proper decode for pandas
            df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')), sep=None, engine='python')

            if 'Class' in df.columns:
                X_input = df.drop('Class', axis=1)
            else:
                X_input = df

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_input)

            predictions = loaded_model.predict(X_scaled)
            statuses = ["Fraudulent" if p == 1 else "Legit" for p in predictions]

            combined_data = [
                {"record": record, "status": status}
                for record, status in zip(df.to_dict(orient='records'), statuses)
            ]

            context = {
                'precision': f'{precision:.2%}',
                'combined_data': combined_data,
            }
            return render(request, 'homeApp/predict_csv_multi.html', context)

        except Exception as e:
            messages.warning(request, f"Invalid/wrong format: {e}")
            return redirect(f'/add_files_multi/{id}')
        
def account_details(request):
    return render(request,'homeApp/account_details.html')
def change_password(request):
    return render(request,'homeApp/change_password.html')
def analysis(request,id):
    obj = DataFileUpload.objects.get(id=id)
    df = load_csv_data(obj.actual_file.path)

    # Empty DataFrame Columns
    empty_columns = len(df.columns[df.isnull().all()].tolist())

    # Data Shape
    data_shape = df.shape

    # Unique Target Values
    # Assuming 'Class' is your target column
    unique_targets = df['Class'].unique().tolist()

    # Percentages
    percent_no_problem = 100 * (df[df['Class'] == 0].shape[0] / df.shape[0])
    percent_problem = 100 * (df[df['Class'] == 1].shape[0] / df.shape[0])

    # Null values check
    has_null = df.isnull().any().any()

    # Transactions
    fraud_transactions = df[df['Class'] == 1]
    normal_transactions = df[df['Class'] == 0]

    fraud_shape = fraud_transactions.shape
    normal_shape = normal_transactions.shape

    # Return or pass these to your template as context
    context = {
        'data_shape': data_shape,
        'unique_targets': unique_targets,
        'percent_no_problem': round(percent_no_problem, 3),
        'percent_problem': round(percent_problem, 3),
        'has_null': has_null,
        'fraud_shape': fraud_shape,
        'normal_shape': normal_shape
    }

    return render(request,'homeApp/analysis.html', context)
def view_data(request,id):
    obj = DataFileUpload.objects.get(id=id)
    df = load_csv_data(obj.actual_file.path)
    columns = df.columns.tolist()
    return render(request,'homeApp/view_data.html', {'id': id, 'columns': columns})
def delete_data(request,id):
    obj=DataFileUpload.objects.get(id=id)
    obj.delete()
    messages.success(request, "File Deleted succesfully",extra_tags = 'alert alert-success alert-dismissible show')
    return HttpResponseRedirect('/reports')
def upload_data(request):
    if request.method == 'POST':
        data_file_name = request.POST.get('data_file_name')
        description = request.POST.get('description')

        try:
            actual_file = request.FILES['actual_file_name']

            if not actual_file.name.endswith('.csv'):
                messages.warning(request, "Please upload a CSV file only.")
                return redirect('/upload_credit_data')

            actual_file.seek(0)
            try:
                data = pd.read_csv(actual_file, sep=None, engine='python')
            except Exception:
                actual_file.seek(0)
                data = pd.read_csv(actual_file, sep=';')

            if data.empty or data.shape[0] < 2:
                messages.warning(request, "Dataset is too small. Please upload at least 2 rows.")
                return redirect('/upload_credit_data')

            if 'Class' not in data.columns:
                messages.warning(request, "Missing 'Class' column (target). Please include it.")
                return redirect('/upload_credit_data')

            # Define features and target
            X = data.drop('Class', axis=1)
            y = data['Class']

            # ✅ handle very small datasets
            test_size = 0.2 if len(X) > 5 else 0.5

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            classifier = LogisticRegression(random_state=0, max_iter=200)
            classifier.fit(X_train_scaled, y_train)

            # Serialize
            serialized_model = pickle.dumps(classifier)
            serialized_x_test = pickle.dumps(X_test_scaled)
            serialized_y_test = pickle.dumps(y_test)

            DataFileUpload.objects.create(
                file_name=data_file_name,
                actual_file=actual_file,
                description=description,
                trained_model_data=serialized_model,
                x_test_data=serialized_x_test,
                y_test_data=serialized_y_test
            )

            messages.success(request, "✅ File uploaded and model trained successfully!")
            return HttpResponseRedirect('/reports')

        except Exception as e:
            messages.warning(request, f"Upload failed: {e}")
            return redirect('/upload_credit_data')






            

def retrieve_data_by_id(request, id):
    obj = DataFileUpload.objects.get(id=id)
    df = load_csv_data(obj.actual_file.path)

    # Receive parameters from DataTables on the frontend
    draw = int(request.GET.get('draw', 1))
    start = int(request.GET.get('start', 0))
    length = int(request.GET.get('length', 10))
    
    # Paginate the data from the CSV using start and length
    paginated_df = df.iloc[start:start+length].reset_index()
    paginated_df['index'] = paginated_df['index'] + 1 + start

    # Convert the paginated data to a list of lists
    data = paginated_df.values.tolist()

    # Return a JSON response suitable for DataTables
    return JsonResponse({
        'draw': draw,
        'recordsTotal': len(df),
        'recordsFiltered': len(df),  # In case you add server-side filtering later on
        'data': data,
    })

def userLogout(request):
    try:
      del request.session['username']
    except:
      pass
    logout(request)
    return HttpResponseRedirect('/') 
    

def login2(request):
    data = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user)
        if user:
            login(request, user)
            return HttpResponseRedirect('/')
        
        else:    
            data['error'] = "Username or Password is incorrect"
            res = render(request, 'homeApp/login.html', data)
            return res
    else:
        return render(request, 'homeApp/login.html', data)


def about(request):
    return render(request,'homeApp/about.html')

def dashboard(request):
    return render(request,'homeApp/dashboard.html')
