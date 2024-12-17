# Add `project_id` to the context if it exists in the session.
def project_id_processor(request):
    return {
        'project_id': request.session.get('project_id', None),
    }