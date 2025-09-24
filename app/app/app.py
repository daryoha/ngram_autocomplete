import reflex as rx
from app.components.autocomplete import autocomplete_input


def index() -> rx.Component:
    """The main page of the app."""
    return rx.el.main(
        rx.el.div(
            rx.el.h1(
                "Carbon Autocomplete",
                class_name="text-3xl font-medium text-gray-100 mb-2",
            ),
            rx.el.p(
                "Start typing to see suggestions.",
                class_name="text-base text-gray-60 mb-8",
            ),
            autocomplete_input(),
            class_name="w-full max-w-2xl p-8 bg-white",
        ),
        class_name="font-['Plex_Sans'] bg-gray-10 flex items-center justify-center min-h-screen",
    )


app = rx.App(
    theme=rx.theme(appearance="light"),
    head_components=[
        rx.el.link(rel="preconnect", href="https://fonts.googleapis.com"),
        rx.el.link(rel="preconnect", href="https://fonts.gstatic.com", cross_origin=""),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap",
            rel="stylesheet",
        ),
    ],
)
app.add_page(index)