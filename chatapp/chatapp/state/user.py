from typing import Dict

import reflex as rx

AUTH = [
    ("admin", "admin"),
]

USERNAMES = [x[0] for x in AUTH]

class User(rx.State):
    logged_in: bool = False
    username: str = ""
    password: str = ""

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

    def set_error(self):
        rx.window_alert("Invalid username or password")
        self.logged_in = False
        return rx.toast.error(
            "Invalid username or password. Please try again!",
            position="top-center",
            duration=4000,
        )

    @rx.event
    def handle_sign_in(self, form_data: Dict[str, str]):
        try:
            idx = USERNAMES.index(self.username)
        except ValueError:
            idx = -1

        if idx != -1:
            password = AUTH[idx][1]
            if password == self.password:
                self.logged_in = True
                return rx.redirect("/")
            else:
                return self.set_error()
        else:
            return self.set_error()

    @rx.event
    def handle_sign_out(self):
        self.reset_state()
        return rx.redirect("/")

    @rx.event
    def reset_state(self):
        self.logged_in = False
        self.username = ""
        self.password = ""

    @rx.var
    def username_display(self) -> str:
        if not self.logged_in:
            return ""

        n = " ".join([x.title() for x in self.username.split("_")]).strip()
        return n


