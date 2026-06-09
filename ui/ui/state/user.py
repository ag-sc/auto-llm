import os

import reflex as rx


import reflex as rx
from passlib.hash import argon2

class UserModel(rx.Model, table=True):
# class UserModel(rx.Model):
    username: str = rx.Field()
    password_hash: str
    is_enabled: bool = True

    @staticmethod
    def hash_password(password: str):
        return argon2.hash(password)

    def verify_password(self, password: str):
        return argon2.verify(password, self.password_hash)


class User(rx.State):
    username: str
    password: str
    logged_in: bool = False

    def handle_sign_up(self):
        with rx.session() as session:
            if self.username is None or self.password is None:
                return rx.toast.error("Please fill the fileds to continue!")

            # Check if user exists
            user = session.exec(UserModel.select().where(UserModel.username == self.username)).first()
            if user:
                return rx.toast.error("Username already exists!")

            # Create and save new user
            new_user = UserModel(username=self.username, password_hash=UserModel.hash_password(self.password))
            session.add(new_user)
            session.commit()
            return rx.toast.success("Registration Successful!")

    def handle_sign_in(self):
        with rx.session() as session:
            user = session.exec(UserModel.select().where(UserModel.username == self.username)).first()
            if user and user.verify_password(self.password):
                self.logged_in = True
                os.environ["autollm_user"] = self.username
                return rx.redirect("/overview")
            else:
                return rx.toast.error("Invalid username or password.")

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
        if not self.logged_in:
            return ""

        n = " ".join([x.title() for x in self.username.split("_")]).strip()
        return n

    @rx.event
    def handle_sign_out(self):
        self.reset_state()
        return rx.redirect("/")

    @rx.event
    def reset_state(self):
        self.logged_in = False
        self.username = ""
        self.password = ""
