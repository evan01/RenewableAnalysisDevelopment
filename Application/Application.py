from flask import Flask, g, jsonify, redirect, request , url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/Application/uploads/'
ALLOWED_EXTENSIONS = set(['csv,xlsx'])

''''''
@app.route('/')
def main():
    '''
    This is loaded first, will load the index.html file
    :return:
    '''
    return redirect(url_for('static',filename="index.html"))

@app.before_request
def before_request():
    print("Before request")

@app.teardown_request
def teardown_request(exception):
    if hasattr(g,"db"):
        g.db.close()

@app.route('/function1')
def function1():
    '''
    Here's an example of a function you can call from the website!
    :return:
    '''
    numItems = request.args.get('nitems',2)
    return jsonify(dict({1: "a",2: "b"}))

@app.route('/',methods=['GET','POST'])
def uploadFile():
    '''
    Uploads a file to the back end for data processing, need to add more security.
    :return:
    '''
    #todo add more security to this
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            #todo write the data processing code for the file

            return redirect(url_for('uploaded_file'),filename=filename)

if __name__ == '__main__':
    # app.run()
    app.run(debug=True,host="localhost", port=5001)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
