"""Bootstrap Galaxy: create admin user and import workflows."""

import json
import os
import sys

from bioblend.galaxy import GalaxyInstance


def get_or_create_admin(gi):
    email = os.environ["GALAXY_ADMIN_EMAIL"]
    username = os.environ["GALAXY_ADMIN_USERNAME"]
    password = os.environ["GALAXY_ADMIN_PASSWORD"]

    for user in gi.users.get_users():
        if user["email"] == email or user["username"] == username:
            print(f"Admin user already exists: {user['username']}")
            return gi.users.get_or_create_user_apikey(user["id"])

    user = gi.users.create_local_user(username, email, password)
    print(f"Created admin user: {username}")
    return gi.users.create_user_apikey(user["id"])


def import_workflows(gi, workflow_dir):
    existing = gi.workflows.get_workflows()
    existing_names = {w["name"] for w in existing}

    for filename in os.listdir(workflow_dir):
        if not filename.endswith(".ga"):
            continue

        path = os.path.join(workflow_dir, filename)
        with open(path) as f:
            name = json.load(f)["name"]

        if name in existing_names:
            old_wf = gi.workflows.get_workflows(name=name)[0]
            gi.workflows.delete_workflow(old_wf["id"])
            print(f"Deleted outdated workflow: {name}")

        wf = gi.workflows.import_workflow_from_local_path(path)
        print(f"Imported workflow: {name}")

        gi.workflows.update_workflow(wf["id"], published=True, menu_entry=True)
        print(f"Published workflow: {name}")


def main():
    url = os.environ["GALAXY_URL"]
    api_key = os.environ["GALAXY_API_KEY"]

    gi = GalaxyInstance(url=url, key=api_key)
    print(f"Connected to Galaxy at {url}")

    user_key = get_or_create_admin(gi)
    gi = GalaxyInstance(url=url, key=user_key)
    print(f"Logged in as: {gi.users.get_current_user()['username']}")

    workflow_dir = "/galaxy-workflows"
    if os.path.isdir(workflow_dir):
        import_workflows(gi, workflow_dir)
    else:
        print(f"No workflow directory found at {workflow_dir}")

    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
