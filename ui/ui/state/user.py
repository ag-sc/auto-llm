import os

import reflex as rx
from passlib.hash import argon2
from sqlmodel import select

# from ..backend.wandb_client import Client
from auto_llm.tracker.tracker import WandbClient

# Users listed here can view and work on behalf of every team (organization)
ADMIN_USERNAMES = {u.strip() for u in os.getenv("ADMIN_USERNAMES", "admin").split(",") if u.strip()}


class UserModel(rx.Model, table=True):
    username: str = rx.Field()
    firstname: str = rx.Field()
    lastname: str = rx.Field()
    organization: str = rx.Field()
    password_hash: str
    is_enabled: bool = True

    @staticmethod
    def hash_password(password: str):
        return argon2.hash(password)

    def verify_password(self, password: str):
        return argon2.verify(password, self.password_hash)


class User(rx.State):
    # Core login tracking states
    username: str = ""
    password: str = ""
    logged_in: bool = False

    user_details: UserModel = None

    # Team (organization) management; only admins can switch away from their own organization
    selected_team: str = ""
    teams: list[str] = []

    # Password management temporary variables
    current_password: str = ""
    new_password: str = ""

    @rx.event
    def handle_key_down(self, key: str):
        # Trigger sign-in only if the user pressed the Enter key
        if key == "Enter":
            return self.handle_sign_in()

    @rx.event
    def handle_sign_up(self):
        with rx.session() as session:
            if not self.username or not self.password:
                return rx.toast.error("Please fill in all fields to continue!")

            # Check if user exists
            user = session.exec(UserModel.select().where(UserModel.username == self.username)).first()
            if user:
                return rx.toast.error("Username already exists!")

            # Create and save new user
            new_user = UserModel(username=self.username, password_hash=UserModel.hash_password(self.password))
            session.add(new_user)
            session.commit()

            # Reset values after registering
            self.password = ""
            return rx.toast.success("Registration Successful!")

    @rx.event
    def handle_sign_in(self):
        with rx.session() as session:
            user = session.exec(UserModel.select().where(UserModel.username == self.username)).first()

            if user and user.verify_password(self.password):
                self.logged_in = True

                self.user_details = user
                self.selected_team = user.organization

                if self.is_admin:
                    organizations = session.exec(select(UserModel.organization).distinct()).all()
                    self.teams = sorted(set(organizations))

                get_wandb_client(user=self, do_login=True)

                # self.password = ""  # Clean sensitive data from frontend memory
                return rx.redirect("/overview")
            else:
                return rx.toast.error("Invalid username or password.")

    @rx.event
    def change_password(self):

        # Ensure user context is present
        if not self.username:
            return rx.toast.error("You must be logged in to change your password.")

        if not self.current_password or not self.new_password:
            return rx.toast.error("Please fill in both fields.")

        with rx.session() as session:
            user = session.exec(UserModel.select().where(UserModel.username == self.username)).first()

            if not user:
                return rx.toast.error("User account could not be found.")

            # Validate original password
            if not user.verify_password(self.current_password):
                return rx.toast.error("Current password incorrect.")

            # Apply and record updates
            user.password_hash = UserModel.hash_password(self.new_password)
            session.add(user)
            session.commit()

            # Clean form elements
            self.current_password = ""
            self.new_password = ""

            return rx.toast.success("Password updated successfully!")

    @rx.event
    def check_login(self):
        if self.logged_in:
            return rx.redirect("/overview")
        return None

    @rx.event
    def check_logged_in(self):
        if not self.logged_in:
            return rx.redirect("/")
        return None

    @rx.var
    def firstname_proc(self) -> str:
        if not self.logged_in or not self.username:
            return ""

        return self.user_details.firstname.strip().capitalize()

    @rx.var
    def lastname_proc(self) -> str:
        if not self.logged_in or not self.username:
            return ""

        return self.user_details.lastname.strip().capitalize()

    @rx.var
    def organization_proc(self) -> str:
        if not self.logged_in or not self.username:
            return ""

        return self.user_details.organization.strip()

    @rx.var
    def is_admin(self) -> bool:
        return self.logged_in and self.username in ADMIN_USERNAMES

    @rx.var
    def active_team(self) -> str:
        if not self.logged_in or not self.username:
            return ""

        if self.is_admin and self.selected_team:
            return self.selected_team.strip()

        return self.organization_proc

    @rx.event
    def set_team(self, team: str):
        if not self.is_admin or team not in self.teams:
            return rx.toast.error("You are not allowed to access this team.")

        self.selected_team = team

        # Imported lazily to avoid circular imports between the user state and the views
        from ..pages.index import OverviewState
        from ..views.workbench_stats import WorkbenchState

        return [OverviewState.fetch_overview_data, WorkbenchState.get_jobs]

    @rx.event
    def handle_sign_out(self):
        self.reset_state()
        return rx.redirect("/")

    @rx.event
    def reset_state(self):
        self.logged_in = False
        self.username = ""
        self.password = ""
        self.current_password = ""
        self.new_password = ""
        self.selected_team = ""
        self.teams = []


def get_wandb_entity(team: str) -> str:
    org = team.replace(" ", "-")
    org = org.lower()
    return f"llm4kmu-{org}"


def get_wandb_client(user: User, do_login: bool = False) -> WandbClient:
    client = WandbClient(entity=get_wandb_entity(user.active_team))

    if do_login:
        client.login(key=os.getenv("WANDB_KEY"))

    return client
