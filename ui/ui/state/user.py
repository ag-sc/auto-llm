import reflex as rx
from passlib.hash import argon2


class UserModel(rx.Model, table=True):
    username: str = rx.Field()
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

    # Password management temporary variables
    current_password: str = ""
    new_password: str = ""

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
    def username_display(self) -> str:
        if not self.logged_in or not self.username:
            return ""
        return " ".join([x.title() for x in self.username.split("_")]).strip()

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
