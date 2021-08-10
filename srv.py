import numpy as np
import os
import joblib
from flask import Flask, request, render_template, make_response, url_for


app = Flask(__name__)
model = joblib.load('/home/gabriel/app_PPT/model/model.pkl')

@app.route('/')
def display_gui():
    return render_template('home.html')

@app.route('/modelo', methods=['POST'])
def carrega_modelo():
    return render_template('template.html')

@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')

@app.route('/about', methods=['POST'])
def sobre():
    return render_template('about.html')

@app.route('/verificar', methods=['POST'])
def verificar():
    partoantptcat   = request.form['partoantptcat']
    ampt            = request.form['ampt']
    qtcpn3t         = request.form['qtcpn3t']
    qtcpn2t         = request.form['qtcpn2t']
    localnp         = request.form['localnp']
    pn              = request.form['pn']
    malcoolg2t      = 0
    causahosp       = request.form['causahosp']
    recanemia       = request.form['recanemia']
    doencagest      = request.form['doencagest']
    hipertgest      = request.form['hipertgest']
    nfilhos         = request.form['nfilhos'] 
    idademae        = request.form['idademae']
    habitofumo      = request.form['habitofumo']
    paroutrab       = request.form['paroutrab']
    fumograv        = request.form['fumograv']
    mes1cpn         = request.form['mes1cpn']
    nuspn           = request.form['nuspn']

    teste = np.array([[partoantptcat, ampt, qtcpn3t, qtcpn2t, localnp, pn, malcoolg2t, 
    causahosp, recanemia, doencagest, hipertgest, nfilhos, idademae, habitofumo,
    paroutrab, fumograv, mes1cpn, nuspn]])

    #print(":::::::::: Dados de Teste :::::::::::")
    #print("Parto ant PT : {}".format(partoantptcat))
    #print("Ameaça parto PT : {}".format(ampt))
    #print("QTD consultas 2T : {}".format(qtcpn2t))
    #print("QTD consultas 3T : {}".format(qtcpn3t))
    #print("Local PN : {}".format(localnp))
    #print("Fez PN : {}".format(pn))
    #print("Média Alcool : {}".format(malcoolg2t))
    #print("Causa Hospitalização : {}".format(causahosp))
    #print("Receitou Anemia : {}".format(recanemia))
    #print("Doenças Gest : {}".format(doencagest))
    #print("Hipert Gestacional : {}".format(hipertgest))
    #print("N Filhos : {}".format(nfilhos))
    #print("Idade Mãe : {}".format(idademae))
    #print("Fumo antes : {}".format(habitofumo))
    #print("Fumo na gestação : {}".format(fumograv))
    #print("Semanas de Trabalho : {}".format(paroutrab))
    #print("Mês inicio PN : {}".format(mes1cpn))
    #print("N US : {}".format(nuspn))
    #print("\n")

    classe = model.predict(teste)[0]
    print("Classe Predita : {}".format(str(classe)))

    return render_template('pred_cl.html', classe =str(classe))

if __name__ == "__main__":
    app.run()
    #port = int(os.environ.get('PORT', 5500))
    #app.run(host='0.0.0.0', port=port, debug=True)

