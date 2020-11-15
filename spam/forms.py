from django import forms

class SearchForm(forms.Form):
    q = forms.CharField(label='',widget=forms.Textarea(
        attrs={
        'class':'search-query form-control',
        'placeholder':'Search'
        }
    ))
