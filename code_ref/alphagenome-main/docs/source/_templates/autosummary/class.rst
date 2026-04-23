{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}


Attributes
~~~~~~~~~~

.. rubric:: Table

.. autosummary::
{% for item in attributes %}
{%- if item not in inherited_members%}
    ~{{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}

{% for item in attributes %}
.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods %}

{% if methods %}

Methods
~~~~~~~

.. rubric:: Table

.. autosummary::
{% for item in methods %}
{%- if item != '__init__' %}
    ~{{ name }}.{{ item }}
{%- endif -%}

{%- endfor %}

{% for item in methods %}
{%- if item != '__init__'%}
.. automethod:: {{ [objname, item] | join(".") }}
{%- endif %}
{%- endfor %}
{% endif %}

{% endblock %}

