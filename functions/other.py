import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
from sklearn.metrics import roc_auc_score, accuracy_score

def print_res_best(res_best, n_features_first=None):
    n_features = len(res_best)
    if n_features_first is None:
        if n_features >= 10:
            print('BEST 10: ')
            for i in range(10):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()
        if n_features >= 50:
            print('BEST 50: ')
            for i in range(50):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()
        if n_features >= 100:
            print('BEST 100: ')
            for i in range(100):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()
    else:
        if n_features >= 10 and n_features_first < 10:
            print('BEST 10: ')
            for i in range(10):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()
        if n_features >= 50 and n_features_first < 50:
            print('BEST 50: ')
            for i in range(50):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()
        if n_features >= 100 and n_features_first < 100:
            print('BEST 100: ')
            for i in range(100):
                print('\t',res_best[i][0], '\t',res_best[i][1])
            print()

def write_dict(res, fold, func_name, model_name, y_test, X_test, y_control, X_control, clf, fillna=False):
    if not fillna:
        res[fold][func_name][model_name]['acc_test'].\
                                    append(accuracy_score(y_test, clf.predict(X_test)))
        res[fold][func_name][model_name]['auc_test'].\
                                    append(roc_auc_score(y_test, clf.predict_proba(X_test).T[1]))

        res[fold][func_name][model_name]['acc_control'].\
                                    append(accuracy_score(y_control, clf.predict(X_control)))
        res[fold][func_name][model_name]['auc_control'].\
                                    append(roc_auc_score(y_control, clf.predict_proba(X_control).T[1]))
    else:
        res[fold][func_name][model_name]['acc_test'].append(float('nan'))
        res[fold][func_name][model_name]['auc_test'].append(float('nan'))
        res[fold][func_name][model_name]['acc_control'].append(float('nan'))
        res[fold][func_name][model_name]['auc_control'].append(float('nan'))
        print(fold, func_name, model_name, 'NaN FILLING')
    return res



def send_mail(send_from, send_to, subject, message, files=[],
              server="smtp.gmail.com", port=587, username='', password='',
              use_tls=True):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="{}"'.format(Path(path).name))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()