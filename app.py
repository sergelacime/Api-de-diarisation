from flask import Flask, render_template, request, jsonify
from diarisation import *
from flask_debugtoolbar import DebugToolbarExtension
app = Flask(__name__)
app.debug = True

# set a 'SECRET_KEY' to enable the Flask session cookies
app.config['SECRET_KEY'] = 'oenzifeeczefzrevrjirefzrejerjferfojrefzkreo'

toolbar = DebugToolbarExtension(app)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():
    # Récupérez les données envoyées par l'utilisateur
    data = request.form.to_dict()
    file = request.files.get('nom')
    speaker = request.files.get('nb')
    if file:
        file.save('media/'+file.filename)
        data['file']=file.filename
    else:
        pass
    # Effectuez le traitement des données
    url_file = 'media/'+file.filename
    # fileurl= speech_to_text(url_file, data['langue'],data['pets2'] , 0)[2]
    # print(speech_to_text(url_file, data['langue'],data['pets2'] , 0)[2])
    # data = readcsv(fileurl)
    result = data  # Ici, nous ne faisons qu'afficher les données envoyées par l'utilisateur
    
    # Renvoyez le résultat au format JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
