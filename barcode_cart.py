# Project Workflow:
# 1. Scan Barcode and obtain product id
# 2. Redirect to Food Cart
# 3. Update the Food Cart using the Quantity(+,-)
# 4. Jump between Pages(Barcode Scan Page and Food Cart Page)
# 4. Add HST Tax(13%) to overall bill
# 5. Create a "Go to Payment" Button to be redirected to Payment Page 
# 6. Add Tip (4 options: 10%, 12%, 15%, 18%)
# 7. Final Bill Amount: Overall Total + Tax + Tip

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pyzbar.pyzbar import decode

# Grocery Dataframe
grocery_df = pd.read_csv(
    "https://raw.githubusercontent.com/selvinfurtado01/schulich_data_science/refs/heads/main/grocery_synthetic_data(details).csv")

# Initialize session state
if "cart" not in st.session_state:
    st.session_state["cart"] = []  # Cart to store products
if "scanned" not in st.session_state:
    st.session_state["scanned"] = False
if "product_id" not in st.session_state:
    st.session_state["product_id"] = None
if "current_screen" not in st.session_state:
    st.session_state["current_screen"] = "scan"  # Default to Scan Page

def main():
    # Toggle between Scan Page, Food Cart, and Payment Page
    if st.session_state["current_screen"] == "scan":
        if st.button("➡️ Go to Food Cart", key="toggle_to_cart"):
            st.session_state["current_screen"] = "cart"
            st.experimental_rerun()
        scan_page()
    elif st.session_state["current_screen"] == "cart":
        if st.button("⬅️ Go to Scan Page", key="toggle_to_scan"):
            st.session_state["current_screen"] = "scan"
            st.experimental_rerun()
        display_cart()
    elif st.session_state["current_screen"] == "payment":
        add_tip_page()

def scan_page():
    st.title("Barcode Scanner")
    st.write("Scan product barcodes to add to the cart!")

    # Checkbox to start/stop webcam
    run = st.checkbox("Scan Code")
    FRAME_WINDOW = st.image([])

    # Open webcam feed
    camera = cv2.VideoCapture(0)
    scanned_product_id = st.empty()

    while run:
        # Capture frame from webcam
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to access the camera! Try Again")
            break

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Decode barcodes in the frame
        decoded_objects = decode(frame)

        for obj in decoded_objects:
            # Extract barcode data (Product ID)
            barcode_data = obj.data.decode("utf-8")
            barcode_type = obj.type

            # Draw rectangle around barcode
            points = obj.polygon
            if len(points) > 4:  # Handles irregular shapes
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull.astype(int).reshape(-1, 2)
                cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=3)

            # Display barcode type and data on frame
            cv2.putText(frame, f"{barcode_type}: {barcode_data}", (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Extract product ID from barcode
            product_id = barcode_data[7:12].strip()  # Adjust slicing based on barcode format
            product_id = int(product_id)
            scanned_product_id.write(f"**Scanned Product ID:** {product_id}")

            # Validate the product ID against the dataset
            if product_id in grocery_df["product_id"].values:
                st.success(f"**Valid Product ID Detected:** {product_id}")

                # Store product ID in session state and redirect
                st.session_state["product_id"] = product_id
                st.session_state["scanned"] = True
                camera.release()  # Release the webcam
                st.experimental_rerun()  # Redirect to food cart
            else:
                st.error("Invalid Product ID! Try again.")
                run = False  # Stop the loop
                break  # Stop processing further in this frame

        # Show the live webcam feed
        FRAME_WINDOW.image(frame)

    camera.release()  # Release webcam when loop exits

def display_cart():
    st.title("Food Cart")
    st.write("Scanned products:")

    # Get scanned product ID
    product_id = st.session_state["product_id"]

    # Lookup product info in the grocery DataFrame
    if product_id is not None:
        product_info = grocery_df[grocery_df["product_id"] == product_id]

        # Add the product to the cart if it's not already in the cart
        if not product_info.empty:
            product = product_info.iloc[0].to_dict()
            product["quantity"] = 1  # Default quantity

            # Check if the product is already in the cart
            if product_id not in [item["product_id"] for item in st.session_state["cart"]]:
                st.session_state["cart"].append(product)

    # Display the cart if it's not empty
    if st.session_state["cart"]:
        # Create and update cart DataFrame
        cart_df = pd.DataFrame(st.session_state["cart"])
        cart_df["total_price"] = cart_df["product_price"] * cart_df["quantity"]

        st.write("Scanned Products:")

        # Display each product with quantity adjustment buttons
        for index, row in cart_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
            col1.write(row["product_name"])
            col2.write(f"Quantity: {row['quantity']}")
            col3.write(f"Price: ${row['product_price']:.2f}")
            col4.write(f"Total: ${row['total_price']:.2f}")

            # Add + and - buttons for quantity adjustment
            if col5.button("➕", key=f"add_{index}"):
                st.session_state["cart"][index]["quantity"] += 1
                st.experimental_rerun()  # Refresh the cart display
            if col5.button("➖", key=f"subtract_{index}"):
                if st.session_state["cart"][index]["quantity"] > 1:
                    st.session_state["cart"][index]["quantity"] -= 1
                else:
                    # Remove item if quantity becomes zero
                    st.session_state["cart"].pop(index)
                st.experimental_rerun()  # Refresh the cart display

        # Update cart DataFrame after any changes
        cart_df = pd.DataFrame(st.session_state["cart"])
        cart_df["total_price"] = cart_df["product_price"] * cart_df["quantity"]

        # Display updated overall total price
        overall_total = cart_df["total_price"].sum()
        tax = overall_total * 0.13  # HST (13%)
        total_with_tax = overall_total + tax

        st.write(f"**Subtotal: ${overall_total:.2f}**")
        st.write(f"**HST (13%): ${tax:.2f}**")
        st.write(f"**Total (Incl. Tax): ${total_with_tax:.2f}**")

        # Store the total with tax for later use in the payment page
        st.session_state["total_with_tax"] = total_with_tax
    else:
        st.write("Your cart is empty!")  # Message when the cart is empty

    # Provide options to clear the cart or proceed to payment
    col1, col2 = st.columns([3, 1])
    if col1.button("Clear Cart"):
        st.session_state["cart"] = []  # Clear all items
        st.session_state["product_id"] = None  # Clear the last scanned product ID
        st.experimental_rerun()
    if col2.button("Go to Payment"):
        st.session_state["current_screen"] = "payment"
        st.experimental_rerun()

def add_tip_page():
    st.title("Add Tip")
    st.write("Choose a tip percentage or enter a custom tip amount.")

    total_with_tax = st.session_state.get("total_with_tax", 0.0)

    # Display tip options and calculate tip amounts
    tip_percentages = [10, 12, 15, 18]
    tip_values = [(p / 100) * total_with_tax for p in tip_percentages]

    col1, col2 = st.columns(2)
    for i, tip in enumerate(tip_percentages):
        with col1 if i % 2 == 0 else col2:
            if st.button(f"{tip}% Tip (${tip_values[i]:.2f})", key=f"tip_{tip}"):
                st.session_state["tip_amount"] = tip_values[i]
                st.experimental_rerun()

    # Custom tip input
    custom_tip = st.number_input("Custom Tip ($)", min_value=0.0, step=0.01)
    if custom_tip > 0:
        if st.button("Apply Custom Tip"):
            st.session_state["tip_amount"] = custom_tip
            st.experimental_rerun()

    # Display total with tip
    tip_amount = st.session_state.get("tip_amount", 0.0)
    final_total = total_with_tax + tip_amount

    # Display Total Price
    st.write(f"**Total Price (incl. Tax): ${total_with_tax:.2f}**")

    # Display Final Total in a bigger font
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 28px; font-weight: bold; margin-top: 20px;">
            Final Total with Tip: ${final_total:.2f}
        </div>
        """,unsafe_allow_html=True,)

    if st.button("Confirm Payment"):
        st.success("Order Placed!")
        # Reset session state
        st.session_state["cart"] = []
        st.session_state["product_id"] = None
        st.session_state["current_screen"] = "scan"
        st.experimental_rerun()

if __name__ == "__main__":
    main()
