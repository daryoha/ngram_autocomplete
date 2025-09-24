import reflex as rx
from app.states.autocomplete_states import AutocompleteState


def autocomplete_input() -> rx.Component:
    """A text input with autocomplete suggestions."""
    return rx.el.div(
        rx.el.div(
            rx.el.input(
                placeholder="Type a something.",
                on_change=AutocompleteState.set_input_text,
                class_name="w-full h-[40px] px-4 text-base text-gray-70 bg-white border border-gray-30 focus:outline-none focus:ring-2 focus:ring-blue-60",
                value=AutocompleteState.input_text,
            ),
            rx.cond(
                AutocompleteState.show_suggestions
                & (AutocompleteState.filtered_items.length() > 0),
                rx.el.div(
                    rx.foreach(
                        AutocompleteState.filtered_items,
                        lambda item: rx.el.div(
                            item,
                            on_click=AutocompleteState.select_item(item),
                            class_name="px-4 py-2 text-gray-70 cursor-pointer hover:bg-gray-10",
                        ),
                    ),
                    class_name="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-20 shadow-lg z-10 max-h-60 overflow-y-auto",
                ),
            ),
            class_name="relative w-full max-w-md"
        ),
        class_name="w-full flex justify-center p-8",
    )