#!/usr/local/bin/python
import cgi
import html
import test_count_cgi as tcc
import count_degree_cgi as cdc

form = cgi.FieldStorage()
text1 = form.getfirst("age", "не задано")
text2 = form.getfirst("bmi", "не задано")
text3 = form.getfirst("volume", "не задано")
text4 = form.getfirst("psa", "не задано")
text1 = html.escape(text1)
text2 = html.escape(text2)
text3 = html.escape(text3)
text4 = html.escape(text4)

res = tcc.get_res(float(text1),float(text2),float(text3),float(text4))
degree = cdc.count_degree(res, text4)

print("Content-type: text/html\n\n")
print("%s " % res[0])
print("%s" % res[1])
print("%s" % res[2])
print("%s " % res[3])
print("%s" % degree)



