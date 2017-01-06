from flask import Flask, g, jsonify, redirect, request , url_for

app = Flask(__name__)


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


if __name__ == '__main__':
    # app.run()
    app.run(debug=True,host="localhost", port=5001)
