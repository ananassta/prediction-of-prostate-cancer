#!/usr/local/bin/python
import cgi
import html

form = cgi.FieldStorage()
text1 = form.getfirst("age", "не задано")
text2 = form.getfirst("bmi", "не задано")
text3 = form.getfirst("volume", "не задано")
text4 = form.getfirst("psa", "не задано")
text1 = html.escape(text1)
text2 = html.escape(text2)
text3 = html.escape(text3)
text4 = html.escape(text4)

print("Content-type: text/html\n\n")
print("%s" % (int(text2)+int(text4)))
