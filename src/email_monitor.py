import os
import argparse
import smtplib
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr


def get_msg(pID, log_path=None):
    res = ""
    status = os.system("ps aux | grep python3 | cut -c 10-15 | grep {:d}".format(pID))
    if status == 0:
        # Don't finish
        return status, res
    res += "<head><style>logStyle {background-color: #ddd;font-size: 20px;}</style></head>"
    res += "<body><h1>{:d} has done!</h1>".format(pID)

    if log_path is not None:
        last_log = open(os.path.join(log_path, "run.log"), "r").readlines()[-1]
        res += "<logStyle>{}</logStyle>".format(last_log)
        res += "<p>Log path: {}</p>".format(log_path)
    res = "<html>" + res + "</html>"
    return status, res


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def main():
    parser = argparse.ArgumentParser("Monitor whether a training task has been completed! Use 'crontab'.")
    parser.add_argument("pID", type=int, help="PID")
    parser.add_argument("--log_path", type=str, help="Log path")
    args = parser.parse_args()

    status, msg = get_msg(args.pID, args.log_path)
    if status == 0:
        return

    # mail settings
    mail_host = 'smtp.163.com'  # SMTP server
    mail_user = 'shanetian'  # Your username
    mail_pass = 'shanetian'  # Your password
    sender = 'shanetian@163.com'  # Your email address
    receivers = ['shanetian1@163.com']  # Target email addresses

    body = MIMEText(msg, "html", "utf-8")
    body["Subject"] = Header("{} has done!".format(args.pID), "utf-8").encode()  # email title
    body["From"] = _format_addr("ECS <{}>".format(sender))  # email sender
    body['To'] = _format_addr("ShaneTian <{}>".format(receivers[0]))  # email receiver

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, body.as_string())
        smtpObj.quit()
    except smtplib.SMTPException as e:
        print('error', e)


if __name__ == "__main__":
    main()