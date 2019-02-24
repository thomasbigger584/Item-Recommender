# django

https://code.visualstudio.com/docs/python/tutorial-django

https://github.com/Microsoft/python-sample-vscode-django-tutorial


# Notes

Use the collectstatic command

For production deployments, you typically collect all the static files from your apps into a single folder using the python manage.py collectstatic command. You can then use a dedicated static file server to serve those files, which typically results in better overall performance. The following steps show how this collection is made, although you don't use the collection when running with the Django development server.

In web_project/settings.py, add the following line that defines a location where static files are collected when you use the collectstatic command:

    STATIC_ROOT = os.path.join(BASE_DIR, 'static_collected')

In the Terminal, run the:

    command python manage.py collectstatic 

and observe that hello/site.css is copied into the top level static_collected folder alongside manage.py.

In practice, run collectstatic any time you change static files and before deploying into production.